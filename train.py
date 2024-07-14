# from envs.funnyworld_cnn import GroundEnv
import copy
import importlib

import torchvision
from torch import nn


from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import torch
import numpy as np
import time

from envs.funnyworld_v5 import *
from algorithm.DDPG import DDPG
from common.tasks import tasks
from common.replay_buffer import ReplayBuffer

# 获取当前脚本所在的项目根目录
PROJECT_PATH = str(os.path.dirname(os.path.abspath(__file__))) #/home/inspur2/workspace/RDDPG


def parse_opt(known=False, algorithm='RDDPG', epoch=200, lra=0.000001, statetype='image3c'):
    parser = argparse.ArgumentParser()
    ##############algorithm
    parser.add_argument('--algorithm', type=str, default=algorithm)
    parser.add_argument('--max_epochs', type=int, default=epoch)
    parser.add_argument('--memory_capacity', type=int, default=2400)
    parser.add_argument('--batchsize', type=int, default=240)
    parser.add_argument('--VAR', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--Vmin', type=int, default=-5)
    parser.add_argument('--Vmax', type=int, default=5)
    parser.add_argument('--lra', type=float, default=lra)
    parser.add_argument('--lrc', type=float, default=0.0001)
    ##############envs
    parser.add_argument('--env', type=str, default='funnyworld_v5')
    parser.add_argument('--render_mode', type=str, default='none')
    parser.add_argument('--actiontype', type=str, default='serial')
    parser.add_argument('--statetype', type=str, default=statetype)
    parser.add_argument('--pos_dim', type=int, default=2)
    ##############others
    parser.add_argument('--task', type=str, default='task_test_model')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--modeldir', type=str, default='models')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def train(opt):
    timestamp = str(int(time.time()))
    logpath = '_'.join([opt.algorithm, timestamp])
    writer = SummaryWriter('log/' + logpath)
    envs = importlib.import_module('envs.' + opt.env)
    task = tasks[opt.task]
    memory = ReplayBuffer(opt.memory_capacity)


    # 4.algorithm
    # 所有的算法遵从以下的形式
    # 4.1生成模型
    # 4.2回合进行，存储状态
    # 4.3模型训练
    if opt.algorithm == 'RDDPG':
        env = envs.GroundEnv(tasks=tasks[opt.task], map=tasks[opt.task]["map"],
                             render_mode=opt.render_mode,
                             label=logpath,
                             actiontype=opt.actiontype, statetype='image3c')  # 适配funnyworld_v5
        s = env.reset()

        if opt.net != '':
            cnnmodel = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            cnnmodel.fc = nn.Sequential()
            cnnmodel.cuda(device=opt.device)
            with torch.no_grad():
                # s = torch.FloatTensor(s).cuda()
                s = torch.FloatTensor(s).unsqueeze(0).cuda()
                s = cnnmodel(s).cpu().numpy()
        # 4.1 model
        VAR = opt.VAR
        REPLACEMENT = [
            dict(name='soft', tau=0.0008),
            dict(name='hard', rep_iter=100)
        ][0]
        model = DDPG(state_dim=s.shape[1],  # todo
                     action_dim=task['radionum'] * opt.pos_dim,
                     replacement=REPLACEMENT,
                     action_bound=1,
                     gamma=opt.gamma,
                     batch_size=opt.batchsize,
                     lr_a=opt.lra,
                     lr_c=opt.lrc, )
        # 4.2acting
        best_epreward = 0
        best_step = 0
        best_model = model
        for i in tqdm(range(1, opt.max_epochs + 1)):
            ep_reward = 0
            for j in range(task['maxstep']):
                a = model.act(s)
                a = np.clip(a + np.clip(np.random.normal(0, VAR, a.shape), -1, 1), -1, 1)  # 在动作选择上添加随机噪声
                # a = np.clip(a , -1, 1)  # 在动作选择上添加随机噪声
                s_, r, done, _, info = env.step(a[0])
                if opt.net != 'none':
                    with torch.no_grad():
                        s_ = torch.FloatTensor(s_).unsqueeze(0).cuda()
                        s_ = cnnmodel(s_).cpu().numpy()
                memory.push(s, a, r, s_, done)
                s = s_

                ep_reward += r
                writer.add_scalar('reward', r, i * task['maxstep'] + j)

                # 4.3
                if len(memory) >= memory._maxsize:
                    loss, q_loss = model.learn(memory, opt.batchsize)
                    writer.add_scalar('td_loss', loss, i * task['maxstep'] + j)
                    writer.add_scalar('q_loss', q_loss, i * task['maxstep'] + j)
                    VAR = max(VAR * 0.998, 0.000001)
                if done:
                    env.reset()
            if ep_reward > best_epreward:
                best_epreward = ep_reward
                best_model = copy.deepcopy(model)
                best_model.save(PROJECT_PATH + '/'+opt.modeldir+'/' + logpath + 'best.pt')
                best_step = i
            writer.add_scalar('ep_reward', ep_reward, i)
        print('best_epreward:', best_epreward)
        model.save(PROJECT_PATH + '/'+opt.modeldir+'/' + logpath + 'final.pt')


    if opt.algorithm == 'DDPG':  # 使用DDPG需要将funnyv5的_get_obs_的生成方式改成mask0
        env = envs.GroundEnv(tasks=tasks[opt.task], map=tasks[opt.task]["map"],
                             render_mode=opt.render_mode,
                             label=logpath,
                             actiontype=opt.actiontype, statetype='vector')  # 适配funnyworld_v5
        s = env.reset()
        VAR = opt.VAR
        REPLACEMENT = [
            dict(name='soft', tau=0.0008),
            dict(name='hard', rep_iter=100)
        ][0]
        model = DDPG(state_dim=s.shape[1],  # todo
                     action_dim=task['radionum'] * opt.pos_dim,
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
                s_, r, done, _, info = env.step(a[0])
                memory.push(s, a, r, s_, done)
                s = s_

                ep_reward += r
                writer.add_scalar('reward', r, i * task['maxstep'] + j)

                # 4.3
                if len(memory) >= memory._maxsize:
                    loss, q_loss = model.learn(memory, opt.batchsize)
                    writer.add_scalar('td_loss', loss, i * task['maxstep'] + j)
                    writer.add_scalar('q_loss', q_loss, i * task['maxstep'] + j)
                    VAR = max(VAR * 0.998, 0.000001)
                if done:
                    env.reset()
            if ep_reward > best_epreward:
                best_epreward = ep_reward
                best_model = copy.deepcopy(model)
                best_model.save(PROJECT_PATH + '/'+opt.modeldir+'/' + logpath + 'best.pt')
                best_step = i
            writer.add_scalar('ep_reward', ep_reward, i)
        print('best_epreward:', best_epreward)
        best_model.save(PROJECT_PATH + '/'+opt.modeldir+'/' + logpath  + 'best.pt')
        model.save(PROJECT_PATH + '/'+opt.modeldir+'/' + logpath + 'final.pt')
    return logpath

if __name__ == '__main__':
    #1.general env
    opt = parse_opt(True, algorithm='RDDPG', epoch=200)
    train(opt)





