import importlib
import os
import random

import numpy as np
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
from algorithm.DDPG import *
from common.tasks import *

import matplotlib.pyplot as plt
import argparse
import time
from tensorboard.backend.event_processing import event_accumulator

PROJECT_PATH = str(os.path.dirname(os.path.abspath(__file__)))
#位置坐标维度(二维世界为x,y, 三维世界为x,y,z)
POS_D = 2
def parse_opt(known=False, algorithm='RDDPG', net='', modelname='', statetype='image3c', task='task_test_model'):
    parser = argparse.ArgumentParser()
    ##############algorithm
    parser.add_argument('--algorithm', type=str, default=algorithm)
    parser.add_argument('--net', type=str, default=net)
    ##############envs
    parser.add_argument('--env', type=str, default='funnyworld_v5')
    parser.add_argument('--render_mode', type=str, default='none')
    parser.add_argument('--actiontype', type=str, default='serial')
    parser.add_argument('--statetype', type=str, default=statetype)
    ##############others
    parser.add_argument('--task', type=str, default=task)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--modeldir', type=str, default='model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--modelname', default=modelname)
    return parser.parse_known_args()[0] if known else parser.parse_args()





def plt2dCurve(data):
    data = np.array(data)
    np.save( PROJECT_PATH+'/analyse/temppos.npy', data)
    radionum = data.shape[1]
    plt.xlim((0,500))
    plt.ylim((-500,0))
    color = ['#0072BD', '#77AC37']
    for i in range(radionum):
        datax = data[:, i, 0]
        datay = data[:, i, 1]
        plt.plot( datay,-datax, c=color[i],linestyle='--')
    plt.show()

def pltClientsRate(opt, data):
    task = tasks[opt.task]


def test(opt):
    timestamp = str(int(time.time()))
    if opt.modelname == '':
        logpath = opt.algorithm + '_' + timestamp
    else:
        logpath = '_'.join([opt.modelname[:-3]])
    writer = SummaryWriter(PROJECT_PATH + '/analyse/' + logpath)
    envs = importlib.import_module('envs.' + opt.env)
    task = tasks[opt.task]
    env = envs.GroundEnv(tasks=tasks[opt.task], map=tasks[opt.task]["map"], render_mode=opt.render_mode, label=logpath,
                         actiontype=opt.actiontype, statetype=opt.statetype)
    s = env.reset()
    # np.save(PROJECT_PATH + "/analyse/s_begin.npy", s)
    # 2.生成模型
    if opt.modelname != '':
        model_dict = torch.load(PROJECT_PATH + '/models/' + opt.modelname)
    if opt.net == 'resnet18':
        cnnmodel = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).cuda()
        cnnmodel.fc = nn.Sequential()
    match opt.algorithm:
        case 'DDPG':
            model = DDPG(model_dict['state_dim'], model_dict['action_dim'], model_dict['action_bound'])
            model.actor.load_state_dict(model_dict['model'])
        case 'RDDPG':
            model = DDPG(model_dict['state_dim'], model_dict['action_dim'], model_dict['action_bound'])
            model.actor.load_state_dict(model_dict['model'])

    # 3.要保存的数据
    radiopos = []
    clients_pos_array = []
    # 4.执行
    ep_reward = 0
    for j in range(task['maxstep']):
        if opt.algorithm == 'withoutRadio':  # 这里应该加一个是连续动作还是离散动作的判断，这里都是离散的
            a = -1 * np.ones(shape=(1, POS_D * task['radionum']))
        if opt.algorithm == 'random':
            a = np.random.random((1, POS_D * task['radionum'])) * 2 - 1
        if opt.algorithm == 'center':
            a = env.updateRadioPosByMeans()
        if opt.algorithm == 'DDPG' and opt.net == '':
            a = model.act(s)
        if opt.net == 'resnet18':
            with torch.no_grad():
                s = cnnmodel(torch.FloatTensor(s).unsqueeze(0).cuda()).cpu().numpy()
                a = model.act(s)
        s_, r, done, _, info = env.step(a)
        s = s_
        ep_reward += r
        writer.add_scalar('reward', r, j)
        cli2base = info[0]
        radiopos.append(info[2])
        clients_pos_array.append(info[3])
        scalar_dict = {}
        for m in range(task['tasknum']):
            scalar_dict['cli' + str(m + 1)] = cli2base[m]
        writer.add_scalars('rate/clients2base', scalar_dict, j)
    radiopos = np.array(radiopos)
    clients_pos_array = np.array(clients_pos_array)
    np.save(PROJECT_PATH + '/analyse/' + logpath + '/radios.npy', radiopos)
    np.save(PROJECT_PATH + '/analyse/' + logpath + '/cli.npy', clients_pos_array)
    print('ep_reward:', ep_reward)
    return logpath



if __name__ == '__main__':
    #1.general env
    opt = parse_opt(True, algorithm='RDDPG', modelname='RDDPG_1720715162final.pt',
                   net='resnet18' )
    # opt = parse_opt(True, algorithm='DDPG', modelname='DDPG_funnyworld_v5_task_epochs800_batchsize488_lra1e-06_lrc0.0001_gamma0.99_VAR3_memory2440_1720624264best.pt'
    #                 , statetype='vector')
    # opt = parse_opt(True, algorithm='withoutRadio')
    test(opt)










