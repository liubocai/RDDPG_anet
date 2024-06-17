#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:27:44
LastEditor: John
LastEditTime: 2022-02-10 01:25:27
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, Categorical
from torch.autograd import Variable
import numpy as np

class MLP(nn.Module):
    
    ''' 多层感知机
        输入：state维度
        输出：概率
    '''
    def __init__(self,input_dim,hidden_dim = 36):
        super(MLP, self).__init__()
        # 24和36为hidden layer的层数，可根据input_dim, n_actions的情况来改变
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 9)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
        
class PolicyGradient:
    
    def __init__(self, n_states, cfg):
        self.gamma = cfg.gamma
        self.policy_net = MLP(n_states,hidden_dim=cfg.hidden_dim)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr) #RMSprop优化器
        self.batch_size = cfg.batch_size

    def choose_action(self,state):

        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)
        # action = torch.multinomial(probs, 1_episodes2000_steps99_var3, replacement=False)
        m = Categorical(probs) # 伯努利分布
        action = m.sample()
        action = int(action.data.numpy()) # 转为标量
        return action
        
    def update(self,reward_pool,state_pool,action_pool):
        # Discount reward
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]  #在一轮游戏中，越到后面所做的决策，*的gamma越多，所占权重越低，后面的决策越不重要
                reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(reward_pool) #这里的reward——pool表示每一个action的奖励
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # Gradient Desent
        self.optimizer.zero_grad()

        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state) #网络推测出的概率
            m = Categorical(probs)
            #action是实际动作当作标签，probs当作网络训练出采取该action的概率，计算这两者的交叉熵作为loss
            loss = -m.log_prob(action) * reward  # Negtive score function x reward   # print(loss) #log_prob()计算的是该取值的概率密度函数的自然对数

            loss.backward()
        self.optimizer.step()
    def save(self,path):
        torch.save(self.policy_net.state_dict(), path+'pg_checkpoint.pt')
    def load(self,path):
        self.policy_net.load_state_dict(torch.load(path+'pg_checkpoint.pt')) 