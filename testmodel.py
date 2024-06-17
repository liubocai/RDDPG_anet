import importlib
import os
import random

import numpy as np
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
from gym_examples.algorithm.CnnRainbowDQN import RainbowCnnDQN
from gym_examples.algorithm.rainbowDQN import RainbowDQN
from gym_examples.algorithm.DDPG import *
from gym_examples.common.tasks import *

import matplotlib.pyplot as plt
import argparse


PROJECT_PATH = '/home/inspur2/workspace/gym-examples/gym_examples/'
#位置坐标维度(二维世界为x,y, 三维世界为x,y,z)
POS_D = 2
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    ##############algorithm
    parser.add_argument('--algorithm', type=str, default='noradio')
    parser.add_argument('--net', type=str, default='')
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
    parser.add_argument('--modelname', default='')
    return parser.parse_known_args()[0] if known else parser.parse_args()



# def oraltasks(map, task, figname):
#     env = GroundEnv(task,map=map, render_mode = 'human')
#     s = env.reset()
#     nodeNum = len(task)
#     maxstep = env.maxstep
#     speedss = []
#     for i in range(nodeNum):
#         speedss.append([])
#     for i in range(1, maxstep+1):
#         a = np.zeros(shape=(2,))
#         s_, r, done, _, info = env.step(a)
#         speeds = info[0]
#         for j in range(nodeNum):
#             speedss[j].append(speeds[j])
#         if done:
#             env.reset()
#     l = [i for i in range(1, maxstep+1)]
#     for i in range(nodeNum):
#         color = tuple(np.random.random(3))
#         plt.plot(l, speedss[i], color=color, label='node'+str(i) if i+1<nodeNum else 'base')
#     plt.legend()
#     plt.title('map:'+map+'  task:'+figname+'  no radio net speed')
#     plt.xlabel('step')
#     plt.ylabel('Mb/s')
#     plt.savefig('../analyse/'+figname+'.jpg')
#     plt.show()
# #
# def pltTaskSpeed(taskn, modeln, env) -> None:
#     envs = importlib.import_module('envs', env)
#     task = tasks[taskn]
#     num = len(task['task'])
#     env = envs.GroundEnv(tasks=tasks[taskn]['task'], map=tasks[taskn]['map'])
#     s = env.reset()
#     #2.如何加载不同的模型并初始化
#     match modeln.split('_')[0]:
#         case 'DDPG':
#             model = genDDPGmodel(modeln)
#         case 'CnnRainbowDQN':
#             model = genCnnRainbowmodel(modeln)
#     model.cuda()
#     #3.完成任务
#     speedss = []
#     for i in range(num):
#         speedss.append([])
#     resnet_model = torchvision.models.resnet18(pretrained=True)
#     resnet_model.fc = nn.Sequential()
#     for i in range(env.maxstep):
#         # s_resnet = resnet_model(s)
#         if modeln.split('_')[0] == "CnnRainbowDQN":
#             a =  model.act(s)
#         else:
#             a = model(s) #todo DDPG+Resnet,需要把resnet也保存下来 #好像用pretrain就可以了
#         s_, r, done, _, info = env.step(a)
#
#         s = s_.copy()
#
#         speeds = list(info[0])
#         for j in range(num):
#             a = speeds[j]
#             speedss[j].append(a)
#         if done:
#             env.reset()
#             break
#     #4.绘图
#     l = [i for i in range(1, len(speedss[0]) + 1)]
#     for j in range(num):
#         plt.plot(l, speedss[j])
#     plt.legend()
#     plt.xlabel('step')
#     plt.ylabel('Mb/s')
#     title = '_'.join([tasks[taskn]['map'] , "task3_224_224_5_2", modeln])
#
#     plt.title('map:' + tasks[taskn]['map'] + '  task:task3_224_224_5_2' + '  al:'+ modeln.split('_')[0])
#     plt.savefig(title+'.jpg')
#     plt.show()
#
#     print(0)
#
# def plt3dCurve(data):
#     data = np.array(data)
#     # np.save('/home/inspur2/workspace/gym-examples/gym_examples/analyse/DDPG_radiopos2.npy', data)
#     radionum = data.shape[1]
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     color = ['#0072BD', '#77AC37']
#     for i in range(radionum):
#         fig = 'figure'+str(i)
#         datax = data[:,i,0]
#         datay = data[:,i,1]
#         dataz = data[:,i,2]
#         fig = ax.plot(datax, datay, dataz, c=color[i],marker='*',linestyle='--')
#     plt.show()

def plt2dCurve(data):
    data = np.array(data)
    np.save('/home/inspur2/workspace/gym-examples/gym_examples/analyse/temppos.npy', data)
    radionum = data.shape[1]
    plt.xlim((0,500))
    plt.ylim((-500,0))
    color = ['#0072BD', '#77AC37']
    for i in range(radionum):
        datax = data[:, i, 0]
        datay = data[:, i, 1]
        plt.plot( datay,-datax, c=color[i],linestyle='--')
    plt.show()




if __name__ == '__main__':
    #1.general env
    opt = parse_opt(True)
    # logpath = '_'.join([opt.modelname[:-3]])
    logpath = 'task13_random_2d_20240401'
    writer = SummaryWriter(PROJECT_PATH + 'analyse/2d/' + logpath)
    envs = importlib.import_module('envs.'+opt.env)
    task = tasks[opt.task]
    # env = envs.GroundEnv(tasks=tasks[opt.task]["task"], map=tasks[opt.task]["map"], render_mode=opt.render_mode, label=logpath, actiontype=opt.actiontype, statetype=opt.statetype)
    env = envs.GroundEnv(tasks=tasks[opt.task], map=tasks[opt.task]["map"], render_mode=opt.render_mode, label=logpath, actiontype=opt.actiontype, statetype=opt.statetype) #适配funnyworld_v4
    s = env.reset()
    np.save(PROJECT_PATH + "analyse/s_begin.npy", s)
    #2.生成模型
    if opt.modelname != '':
        model_dict = torch.load(PROJECT_PATH + 'models/2d/'+opt.modelname)
    if opt.net == 'resnet18':
        cnnmodel = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).cuda()
        cnnmodel.fc = nn.Sequential()
    match opt.algorithm:
        case 'RainbowDQN':
            model = RainbowDQN(model_dict['num_inputs'],model_dict['num_actions'],model_dict['num_atoms'],model_dict['Vmin'],model_dict['Vmax']).cuda()
            model.load_state_dict(model_dict['model'])
        case 'DDPG':
            model = DDPG(model_dict['state_dim'],model_dict['action_dim'],model_dict['action_bound'])
            model.actor.load_state_dict(model_dict['model'])

    #3.要保存的数据
    radiopos = []
    clients_pos_array = []
    #4.执行
    ep_reward = 0
    for j in range(task['maxstep']):
        if opt.algorithm == 'noradio': #这里应该加一个是连续动作还是离散动作的判断，这里都是离散的
            a = np.zeros(shape=(1, POS_D * task['radionum']))
        if opt.algorithm == 'random':
            a = np.random.random((1, POS_D * task['radionum'])) * 2 - 1
        if opt.algorithm == 'mean':
            a = env.updateRadioPosByMeans()
        if opt.algorithm == 'DDPG' and opt.net == '':
            a = model.act(s)
            print(str(j), ': ', a)
        if opt.net == 'resnet18':
            with torch.no_grad():
                s = cnnmodel(torch.FloatTensor(s).unsqueeze(0).cuda()).cpu().numpy()
                a = model.act(s)
                print(str(j), ': ', a)
        s_, r, done, _, info = env.step(a)
        s = s_
        ep_reward += r
        writer.add_scalar('reward',r,j)
        cli2base = info[0]
        radiopos.append(info[2])
        clients_pos_array.append(info[3])
        scalar_dict = {}
        for m in range(task['tasknum']):
            scalar_dict['cli'+str(m+1)] = cli2base[m]
        writer.add_scalars('speeds/clients2base', scalar_dict, j)
    radiopos = np.array(radiopos)
    clients_pos_array = np.array(clients_pos_array)
    np.save(PROJECT_PATH + 'analyse/2d/' + logpath + '.npy' , radiopos)
    np.save(PROJECT_PATH + 'analyse/random/' + logpath + 'cli_2.npy' , clients_pos_array)
    print('ep_reward:', ep_reward)







