import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
from matplotlib import cm
from matplotlib.colors import LightSource
import pandas as pd

CMAX = 41.2771
COLORS = ['#EDB120', '#7E2F8E', '#77AC30', '#0072BD']
PATH = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLORDICT = {
    'WithoutRadio': '#EDB120',
    'Centerlized': '#7E2F8E',
    'DDPG': '#77AC30',
    'RDDPG': '#0072BD',
}
ALGORITHM = ['WithoutRadio', 'Centerlized', 'DDPG', 'RDDPG']
def drawTrainingReward(logdir, label):
    logdir = PATH +'/log/' + logdir
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    tag = 'ep_reward'
    events = ea.Scalars(tag)
    # 提取时间戳和数值
    steps = [event.step for event in events]
    values = [event.value for event in events]

    # 使用matplotlib绘制数据
    plt.figure()
    plt.plot(steps, values, label=tag)
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.title('Training reward of ' + label)
    plt.legend()
    plt.show()

def drawClientsRateChangeLine(logdir):
    logdir = PATH + '/analyse/' + logdir
    entries = os.listdir(logdir)
    clients = 0
    for entry in entries:
        if os.path.isdir(os.path.join(logdir, entry)):
            clients += 1
    # fig, ax = plt.subplots(1,clients)
    for i in range(clients):
        cliLogDir = os.path.join(logdir, 'rate_clients2base_cli'+str(i+1))
        ea = event_accumulator.EventAccumulator(cliLogDir)
        ea.Reload()
        events = ea.Scalars('rate/clients2base')
        values = [CMAX* event.value for event in events]
        steps = [event.step for event in events]
        plt.plot(steps, values, label='client'+str(i+1))
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Communication rate (Mb/s)')
    plt.title('Communication rate variation graph')
    plt.show()
        # axs[int(i / 2)][i % 2].plot(np.arange(len(d3)), d3, color='#EDB120', label='Wi   thout        RANET        node        ')
        # axs[int(i / 2)][i % 2].plot(np.arange(len(d4)), d4, color='#7E2F8E', label='Ce        ntralized       deployment        ')
        # axs[int(i / 2)][i % 2].plot(np.arange(len(d2)), d2, color='#77AC30', label='D        DPG        ')
        # axs[int(i / 2)][i % 2].plot(np.arange(len(d1)), d1, color='#0072BD', label='R        DDPG        ')
        #
        # axs[int(i / 2)][i % 2].set_title('Real-time communication rate of node' + str(i + 1
        #                                                                               ), fontsize=8)
        # axs[int(i / 2)][i % 2].set_xlabel('Time step')
        # axs[int(i / 2)][i % 2].set_ylabel('Communication rate (Mb/s)')
        # axs[int(i / 2)][i % 2].xaxis.label.set_fontsize(8)
        # axs[int(i / 2)][i % 2].yaxis.label.set_fontsize(8)

def pltDataFromTensorboard(logdir, tag):
    # TensorBoard事件文件的路径

    # 创建一个事件累加器，读取事件文件
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    # tags = ['ep_reward']

    # 获取该标签的所有数据
    events = ea.Scalars(tag)
    # 提取时间戳和数值
    timestamps = [event.wall_time for event in events]
    values = [event.value for event in events]

    # 使用matplotlib绘制数据
    plt.figure()
    plt.plot(timestamps, values, label=tag)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('TensorBoard Data for ' + tag)
    plt.legend()
    plt.show()

def drawRadiosTrace(log):
    radioNpypath = PATH + '/analyse/' + log + '/radios.npy'
    clientsNpyPath = PATH + '/analyse/' + log + '/cli.npy'
    data_radio = np.load(radioNpypath)
    data_clients = np.load(clientsNpyPath)

    fig = plt.figure()
    backgroundImg = plt.imread(PATH+'/map/dom.png')
    plt.imshow(backgroundImg, extent=[0, 500, -500, 0])
    plt.xlim((0, 500))
    plt.ylim((-500, 0))
    radionum = data_radio.shape[1]
    clinum = data_clients.shape[1]

    for j in range(clinum):
        clix = data_clients[:, j, 0]
        cliy = data_clients[:, j, 1]
        plt.plot(cliy, -clix, c=COLORDICT['WithoutRadio'], linestyle='dashed', label='sensing paths')
    for j in range(radionum):
        radiox = data_radio[:, j, 0]
        radioy = data_radio[:, j, 1]
        plt.plot(radioy, -radiox)
    # 去除重复标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    unique_labels = set(labels)
    unique_handles = [by_label[label] for label in unique_labels]

    # 创建图例
    plt.legend(unique_handles, unique_labels)

    # plt.legend()
    # print(radios)
    # fig.savefig('analyse/paths_true_618.png')
    plt.show()

def drawClientsRateCompare(loglist):
    logdir = PATH + '/analyse/' + loglist[0]
    entries = os.listdir(logdir)
    clients = 0
    for entry in entries:
        if os.path.isdir(os.path.join(logdir, entry)):
            clients += 1
    fig, ax = plt.subplots(1,clients, figsize=(16,4))
    for i in range(clients):
        for j in range(len(loglist)):
            cliLogDir = os.path.join(PATH+'/analyse/'+loglist[j], 'rate_clients2base_cli'+str(i+1))
            ea = event_accumulator.EventAccumulator(cliLogDir)
            ea.Reload()
            events = ea.Scalars('rate/clients2base')
            values = [CMAX* event.value for event in events]
            steps = [event.step for event in events]
            ax[i].plot(steps, values, color=COLORS[j], label=ALGORITHM[j])
        # ax[i].set_title('Real-time communication rate of node' + str(i + 1), fontsize='small')
        ax[i].set_xlabel('Time step\n' + 'node' +str(i+1))
        ax[i].set_ylabel('Communication rate (Mb/s)')
    plt.title('Real-time communication rate of all nodes')
    # 去除重复标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    unique_labels = set(labels)
    unique_handles = [by_label[label] for label in unique_labels]

    # 创建图例
    plt.legend(unique_handles, unique_labels)
    plt.show()


def drawRadiosTraceCompare(loglist):
    fig = plt.figure()
    backgroundImg = plt.imread(PATH+'/map/dom.png')
    plt.imshow(backgroundImg, extent=[0, 500, -500, 0])
    plt.xlim((0, 500))
    plt.ylim((-500, 0))

    for i in range(len(loglist)):
        radioNpypath = PATH + '/analyse/' + loglist[i] + '/radios.npy'
        clientsNpyPath = PATH + '/analyse/' + loglist[i] + '/cli.npy'
        data_radio = np.load(radioNpypath)
        data_clients = np.load(clientsNpyPath)
        radionum = data_radio.shape[1]
        clinum = data_clients.shape[1]
        # trace of clients only to be drwan once
        if i==0:
            for j in range(clinum):
                clix = data_clients[:, j, 0]
                cliy = data_clients[:, j, 1]
                plt.plot(cliy, -clix, c=COLORDICT['WithoutRadio'], linestyle='dashed', label='sensing paths')
        for j in range(radionum):
            radiox = data_radio[:, j, 0]
            radioy = data_radio[:, j, 1]
            plt.plot(radioy, -radiox, label=ALGORITHM[i+1], color=COLORS[i+1])
    # 去除重复标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    unique_labels = set(labels)
    unique_handles = [by_label[label] for label in unique_labels]

    # 创建图例
    plt.legend(unique_handles, unique_labels)
    plt.title('RANET nodes motion trajectory graph')
    plt.show()

def calMeanRate(loglist):
    logdir = PATH + '/analyse/' + loglist[0]
    entries = os.listdir(logdir)
    clients = 0
    for entry in entries:
        if os.path.isdir(os.path.join(logdir, entry)):
            clients += 1
    meanRate = np.empty(shape=(clients, len(loglist)))
    for i in range(clients):
        for j in range(len(loglist)):
            cliLogDir = os.path.join(PATH+'/analyse/'+loglist[j], 'rate_clients2base_cli'+str(i+1))
            ea = event_accumulator.EventAccumulator(cliLogDir)
            ea.Reload()
            events = ea.Scalars('rate/clients2base')
            values = [CMAX* event.value for event in events]
            meanRate[i, j] = np.mean(values)
    return meanRate

if __name__ == '__main__':
    log = 'RDDPG_1720439330'
    drawTrainingReward(log, 'RDDPG')
    drawClientsRateChangeLine(log)
    drawRadiosTrace(log)
