# from envs.funnyworld_cnn import GroundEnv
import copy
import importlib

import torchvision
from torch import nn


from algorithm.rainbowDQN import RainbowDQN
from algorithm.PPO import PPO
from torch.utils.tensorboard import SummaryWriter
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm
import argparse
import torch
import numpy as np

from gym_examples.algorithm.DDPG import DDPG
from gym_examples.common.tasks import tasks
from gym_examples.common.replay_buffer import ReplayBuffer
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
#todo 1.保存训练后的模型
#todo 2.任务的生成需要更符合实际
#todo 3.使用CnnDQN
#todo 4.建筑物的碰撞约束
#todo 5.其他终端的速度

PROJECT_PATH = '/home/inspur2/workspace/gym-examples/gym_examples/'
#位置坐标维度(二维世界为x,y, 三维世界为x,y,z)
POS_D = 2
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    ##############algorithm
    parser.add_argument('--algorithm', type=str, default='DDPG')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--memory_capacity', type=int, default=2440)
    parser.add_argument('--batchsize', type=int, default=488)
    parser.add_argument('--VAR', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--net', type=str, default='')
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--Vmin', type=int, default=-5)
    parser.add_argument('--Vmax', type=int, default=5)
    parser.add_argument('--lra', type=float, default=0.000001)
    parser.add_argument('--lrc', type=float, default=0.0001)
    ##############envs
    parser.add_argument('--env', type=str, default='funnyworld_v5')
    parser.add_argument('--render_mode', type=str, default='none')
    parser.add_argument('--actiontype', type=str, default='serial')
    parser.add_argument('--statetype', type=str, default='image3c')
    ##############others
    parser.add_argument('--task', type=str, default='task13_500_4_2_1_3D')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--modeldir', type=str, default='model')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def projection_distribution(next_state, rewards, dones):
    batch_size = next_state.size(0)

    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms) #从vmin到vmax平分成num_atoms份

    next_dist = target_model(next_state).data.cpu() * support #(batch, out, num_atoms) 把网络输出的9个q值乘以上述的51份间隔
    next_action = next_dist.sum(2).max(1)[1] #取1是因为max函数0是原来的值，1表示对应的索引
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2)) #升维，原来的一个动作复制成num——atoms份
    next_dist = next_dist.gather(1, next_action).squeeze(1) #（batch4,atom_nums51）

    rewards = rewards.unsqueeze(1).expand_as(next_dist) #（4,51）
    dones = dones.unsqueeze(1).expand_as(next_dist) #（4,51）
    support = support.unsqueeze(0).expand_as(next_dist)#（4,51）

    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b = (Tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long() \
        .unsqueeze(1).expand(batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    return proj_dist


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = memory.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    with torch.no_grad():
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(np.float32(done))

    proj_dist = projection_distribution(next_state, reward, done)

    dist = model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, action).squeeze(1)  #gather:
    dist.data.clamp_(0.01, 0.99)
    loss = -(Variable(proj_dist) * dist.log()).sum(1)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.reset_noise()
    target_model.reset_noise()

    return loss



if __name__ == '__main__':
    #1.general env
    opt = parse_opt(True)
    logpath = '_'.join([opt.algorithm, opt.env, opt.task.split('_')[0] ,
                        'epochs'+str(opt.max_epochs), 'batchsize'+str(opt.batchsize), 'lra'+str(opt.lra), 'lrc'+str(opt.lrc), 'gamma'+str(opt.gamma), 'VAR'+str(opt.VAR), 'memory'+str(opt.memory_capacity),
                        'rewardp_delmin_02211800'])
    writer = SummaryWriter('log/2d/'+logpath)
    envs = importlib.import_module('envs.'+opt.env)
    task = tasks[opt.task]
    # env = envs.GroundEnv(tasks=tasks[opt.task]["task"], map=tasks[opt.task]["map"], render_mode=opt.render_mode, label=logpath, actiontype=opt.actiontype, statetype=opt.statetype)
    env = envs.GroundEnv(tasks=tasks[opt.task], map=tasks[opt.task]["map"], render_mode=opt.render_mode, label=logpath, actiontype=opt.actiontype, statetype=opt.statetype) #适配funnyworld_v5
    s = env.reset()
    #2.cnnmodel
    match opt.net:
        case 'resnet18':
            cnnmodel = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            cnnmodel.fc = nn.Sequential()
            cnnmodel.cuda(device=opt.device)
    if opt.net != '':
        with torch.no_grad():
            # s = torch.FloatTensor(s).cuda()
            s = torch.FloatTensor(s).unsqueeze(0).cuda()
            s = cnnmodel(s).cpu().numpy()
    #3.replay buffer
    memory = ReplayBuffer(opt.memory_capacity)
    #4.algorithm
    #所有的算法遵从以下的形式
    #4.1生成模型
    #4.2回合进行，存储状态
    #4.3模型训练
    if opt.algorithm == 'Resnet_DDPG':
        #4.1 model
        VAR = opt.VAR
        REPLACEMENT = [
            dict(name='soft', tau=0.0008),
            dict(name='hard', rep_iter=100)
        ][0]
        model = DDPG(state_dim=s.shape[1],  # todo
                    action_dim=task['radionum'] * POS_D,
                    replacement=REPLACEMENT,
                    action_bound=1,
                    gamma=opt.gamma,
                    batch_size=opt.batchsize,
                    lr_a = opt.lra,
                    lr_c = opt.lrc,)
        #4.2acting
        best_epreward = 0
        best_step = 0
        best_model = model
        for i in tqdm(range(1, opt.max_epochs+1)):
            ep_reward = 0
            # for j in range(task['maxstep']):
            for j in range(20000):
                a = model.act(s)
                a = np.clip(a + np.clip(np.random.normal(0, VAR, a.shape), -1, 1), -1, 1)  # 在动作选择上添加随机噪声
                # a = np.reshape(a[0], (POS_D, task['radionum'])) #改变结构
                s_, r, done, _, info = env.step(a[0])
                if opt.net != 'none':
                    with torch.no_grad():
                        s_ = torch.FloatTensor(s_).unsqueeze(0).cuda()
                        s_ = cnnmodel(s_).cpu().numpy()
                memory.push(s,a,r,s_,done)
                s = s_

                ep_reward += r
                writer.add_scalar('reward', r, i*task['maxstep']+j)

                #4.3
                if len(memory) >= memory._maxsize:
                    loss = model.learn(memory, opt.batchsize)
                    writer.add_scalar('loss', loss, i*task['maxstep']+j)
                    VAR = max(VAR*0.998, 0.000001)
                if done:
                    env.reset()
            if ep_reward > best_epreward:
                best_epreward = ep_reward
                best_model = copy.deepcopy(model)
                best_model.save(PROJECT_PATH + 'models/2d/' + logpath + 'best.pt')
                best_step = i
            # if i % 200 ==0:
            #     best_model.save(PROJECT_PATH + 'models/2d/' + logpath + 'best.pt')
            #     model.save(PROJECT_PATH + 'models/2d/' + logpath + 'final.pt')
            writer.add_scalar('ep_reward', ep_reward, i)
        print('best_epreward:',best_epreward, "  appears in:", best_step)
        model.save(PROJECT_PATH + 'models/2d/' + logpath + 'final.pt')
        print('var:',VAR)

    if opt.algorithm == 'DDPG': #使用DDPG需要将funnyv5的_get_obs_的生成方式改成mask0
        VAR = opt.VAR
        REPLACEMENT = [
            dict(name='soft', tau=0.0008),
            dict(name='hard', rep_iter=100)
        ][0]
        model = DDPG(state_dim=s.shape[1],  # todo
                     action_dim=task['radionum'] * POS_D,
                     replacement=REPLACEMENT,
                     action_bound=1,
                     gamma=opt.gamma,
                     batch_size=opt.batchsize,
                     lr_a=opt.lra)
        # 4.2acting
        best_epreward = 0
        best_model = model
        for i in tqdm(range(1, opt.max_epochs + 1)):
            ep_reward = 0
            for j in range(task['maxstep']):
                a = model.act(s)
                a = np.clip(a + np.clip(np.random.normal(0, VAR, a.shape), -1, 1), -1, 1)  # 在动作选择上添加随机噪声
                # a = np.reshape(a[0], (POS_D, task['radionum'])) #改变结构
                s_, r, done, _, info = env.step(a[0])
                if opt.net != '':
                    with torch.no_grad():
                        s_ = torch.FloatTensor(s_).unsqueeze(0).cuda()
                        s_ = cnnmodel(s_).cpu().numpy()
                memory.push(s, a, r, s_, done)
                s = s_

                ep_reward += r
                writer.add_scalar('reward', r, i * task['maxstep'] + j)

                # 4.3
                if len(memory) >= memory._maxsize:
                    loss = model.learn(memory, opt.batchsize)
                    writer.add_scalar('loss', loss, i * task['maxstep'] + j)
                    VAR = max(VAR * 0.998, 0.000001)
                if done:
                    env.reset()
            if ep_reward > best_epreward:
                best_epreward = ep_reward
                best_model = copy.deepcopy(model)
                best_model.save(PROJECT_PATH + 'models/2d/' + logpath + 'best.pt')
                best_step = i
            writer.add_scalar('ep_reward', ep_reward, i)
        print('best_epreward:', best_epreward, "  appears in:", best_step)
        best_model.save(PROJECT_PATH + 'models/2d/' + logpath + 'best.pt')
        model.save(PROJECT_PATH + 'models/2d/' + logpath + 'final.pt')


    if opt.algorithm == 'Resnet_RainbowDQN':

        #4.1 model
        num_atoms = opt.num_atoms
        Vmin = opt.Vmin
        Vmax = opt.Vmax
        model = RainbowDQN(s.shape[1], env.action_space.n, opt.num_atoms, opt.Vmin, opt.Vmax)
        target_model = RainbowDQN(s.shape[1], env.action_space.n, opt.num_atoms, opt.Vmin, opt.Vmax)
        model = model.cuda()
        target_model = target_model.cuda()
        optimizer = optim.Adam(model.parameters(), opt.lr)
        #4.2
        best_epreward = 0
        best_model = model
        for i in tqdm(range(1, opt.max_epochs+1)):
            ep_reward = 0
            update_target(model, target_model)
            for j in range(task['maxstep']):
                a = model.act(s)
                s_, r, done, _, info = env.step(a)
                if opt.net != 'none':
                    with torch.no_grad():
                        s_ = torch.FloatTensor(s_).unsqueeze(0).cuda()
                        s_ = cnnmodel(s_).cpu().numpy()
                memory.push(s,a,r,s_,done)
                s = s_

                ep_reward += r
                writer.add_scalar('reward', r, i*task['maxstep']+j)
                #4.3
                if len(memory) >= memory._maxsize:
                    loss = compute_td_loss(opt.batchsize)
                    writer.add_scalar('loss', loss, i*task['maxstep']+j)
                if done:
                    env.reset()
            if ep_reward > best_epreward:
                best_epreward = ep_reward
                best_model = copy.deepcopy(model)
            writer.add_scalar('ep_reward', ep_reward, i)
        print('best_epreward:',best_epreward)
        best_model.save(PROJECT_PATH+'models/'+logpath+'best.pt')
        model.save(PROJECT_PATH+'models/'+logpath+'final.pt')

    if opt.algorithm == 'PPO':
        K_epochs = 80
        eps_clip = 0.2
        if 'serial' == opt.actiontype:
            has_continuous_action_space = True
        else:
            has_continuous_action_space = False
        model = PPO(state_dim=s.shape[1], action_dim=task['radionum'] * POS_D,
                    lr_actor=opt.lra,
                    lr_critic=opt.lrc,
                    gamma=opt.gamma,
                    K_epochs=K_epochs,
                    eps_clip=eps_clip,
                    has_continuous_action_space=has_continuous_action_space)
        best_epreward = 0
        best_model = model
        for i in tqdm(range(1, opt.max_epochs + 1)):
            ep_reward = 0
            for j in range(task['maxstep']):
                a = model.select_action(s)
                s_, r, done, _, info = env.step(a[0])
                model.buffer.rewards.append(r)
                model.buffer.is_terminals.append(done)
                if opt.net != 'none':
                    with torch.no_grad():
                        s_ = torch.FloatTensor(s_).unsqueeze(0).cuda()
                        s_ = cnnmodel(s_).cpu().numpy()
                s = s_

                ep_reward += r
                writer.add_scalar('reward', r, i * task['maxstep'] + j)
                # 4.3
                if done:
                    env.reset()
            if ep_reward > best_epreward:
                best_epreward = ep_reward
                best_model = copy.deepcopy(model)
            writer.add_scalar('ep_reward', ep_reward, i)

            # learn
            if i % opt.batchsize == 0:
                loss = model.update()
                writer.add_scalar('loss', loss, i / opt.batchsize)
        print('best_epreward:', best_epreward)
        best_model.save(PROJECT_PATH + 'models/' + logpath + 'best.pt')
        model.save(PROJECT_PATH + 'models/' + logpath + 'final.pt')
