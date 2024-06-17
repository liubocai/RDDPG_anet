from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
label = 'Resnet_DDPG_funnyworld_v5_task13_epochs200_batchsize488_lra1e-06_lrc0.0001_gamma0.99_VAR3_memory2440_rewardp_num2_delmin_2best'
# label = 'Resnet_DDPG_funnyworld_v5_task13_epochs200_batchsize488_lra1e-06_lrc0.0001_gamma0.99_VAR3_memory2440_num2_rewardpfinal'
# label = 'Resnet_DDPG_funnyworld_v5_task13_epochs200_batchsize488_lra1e-06_lrc0.0001_gamma0.99_VAR3_memory2440_rewardp_num2_delmin_3best'
label2 = 'DDPG_funnyworld_v5_task13_epochs200_batchsize488_lra1e-06_lrc0.0001_gamma0.99_VAR3_memory2440_rewardp_02211600final'
label3 = 'task13_random_2d'

label4 = 'DDPG_funnyworld_v5_task13_epochs1000_batchsize256_lra1e-06_lrc0.0001_gamma0.99_VAR2best'
#1.读数据
data = np.load('analyse/2d/'+label+'.npy')
data_DDPG = np.load('analyse/2d/'+label2+'.npy')
data_center = np.load('analyse/2d/'+label3+'.npy')
data_mid = np.load('analyse/2d/'+label4+'.npy')

ddpg1 = pd.read_excel('analyse/output_ddpg1.xlsx')
ddpg1 = ddpg1.to_numpy()
ddpg2 = pd.read_excel('analyse/output_ddpg2.xlsx')
ddpg2 = ddpg2.to_numpy()
rddpg1 = pd.read_excel('analyse/output_rddpg1.xlsx')
rddpg1 = rddpg1.to_numpy()
rddpg2 = pd.read_excel('analyse/output_rddpg2.xlsx')
rddpg2 = rddpg2.to_numpy()

p = 'gym_examples/analyse/random/task13_random_2d_20240401cli_2.npy'
data_clis = np.load('analyse/random/task13_random_2d_20240401cli_2.npy')
# data_cli3 = data_clis[:,3,:]
# np.savetxt('analyse/cli4_path.txt', data_cli3)
radionum = data.shape[1]
clinum = data_clis.shape[1]
clis = data_clis
radios = data

compare_path = np.mean(data_clis, axis=1)
print(compare_path.shape)
#2.绘图

color = {
    'base': '#EDB120',
    'RDDPG': '#0072BD',
    'DDPG': '#77AC30',
    'Center': '#7E2F8E'
}

fig = plt.figure(dpi=1000)
ax = fig.add_subplot(1,1,1)
img = plt.imread('map/b500_.png')
plt.imshow(img, extent=[0,500,-500,0])
# ax.imshow('map/b500_.png')
plt.xlim((0,500))
plt.ylim((-500,0))
# color = [ '#0072BD', '#77AC37','r', 'b']
marker = ['*', 'o']
# # 画任务轨迹
# task1 = plt.Rectangle((90, -310), 170,200, edgecolor='black', fill=False)
# plt.text(90, -310, 'Task1')
# task2 = plt.Rectangle((280, -190), 100,50, edgecolor='black', fill=False)
# plt.text(280, -190, 'Task2')
# task3 = plt.Rectangle((110, -410), 140, 90, edgecolor='black', fill=False)
# plt.text(110, -410, 'Task3')
# task4 = plt.Rectangle((380, -420), 100,160, edgecolor='black', fill=False)
# plt.text(380, -420, 'Task4')
# ax.add_patch(task1)
# ax.add_patch(task2)
# ax.add_patch(task3)
# ax.add_patch(task4)
plt.plot( compare_path[:,1],-compare_path[:,0], c=color['Center'], linestyle='solid', label = 'Center')  #绘制cluster的路径，中间一坨，但不知道线的走向
for j in range(clinum):
    clix = clis[:, j, 0]
    cliy = clis[:, j, 1]
    plt.plot( cliy,-clix, c=color['base'], linestyle='dashed', label='sensing paths')
plt.plot(ddpg1[:,1], -ddpg1[:,0], color=color['DDPG'], label='DDPG')
plt.plot(ddpg2[:,1]-3, -ddpg2[:,0], color=color['DDPG'], label='DDPG')
plt.plot(rddpg1[:,1], -rddpg1[:,0], color=color['RDDPG'], label='RDDPG')
plt.plot(rddpg2[:,1], -rddpg2[:,0], color=color['RDDPG'], label='RDDPG')

# for i in range(radionum):
#     datax = data[:, i, 0] + int(i) *2
#     datay = data[:, i, 1]+ int(i)
#     datax2 = data_DDPG[:, i, 0] + int(i) *2
#     datay2 = data_DDPG[:, i, 1]+ int(i)
# 
#     plt.plot( datay,-datax, c=color['RDDPG'], linestyle='solid', label='RDDPG')
    # plt.plot( datay2,-datax2, c=color['DDPG'], linestyle='solid', label='DDPG')
    # plt.plot( datay3,-datax3, c=color['Center'], linestyle='solid', label='Center')
    # dotx = radios[:, i, 0] + int(i)*2
    # doty = radios[:, i, 1]+ int(i)
    # plt.scatter( doty,-dotx, c=color[i], marker=marker[i])

    # plt.scatter( cliy,-clix, c='b', marker='o')
    # print(clix, cliy)

# 去除重复标签
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
unique_labels = set(labels)
unique_handles = [by_label[label] for label in unique_labels]

# 创建图例
plt.legend(unique_handles, unique_labels)

# plt.legend()
# print(radios)
fig.savefig('analyse/paths_true_412.png')
plt.show()