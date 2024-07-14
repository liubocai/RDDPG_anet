"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
torch实现DDPG算法
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models
import torch.autograd


'''
seed = 1_episodes2000_steps99_var3
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)
'''
# 定义训练的设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

point=0
best=-100000
best_episode=0
best_action=[]
best_data=[]
# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = torch.FloatTensor(action_bound).to(device)

        # layer
        self.layer_1 = nn.Linear(state_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        # nn.init.normal_(self.layer_1.weight, 0., 0.3)
        # nn.init.constant_(self.layer_1.bias, 0.1_episodes2000_steps99_var3)
        # self.layer_1.weight.data.normal_(0.,0.3)
        # self.layer_1.bias.data.fill_(0.1_episodes2000_steps99_var3)
        self.layer_2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        # nn.init.normal_(self.layer_2.weight, 0., 0.3)
        # nn.init.constant_(self.layer_2.bias, 0.1_episodes2000_steps99_var3)

        self.layer_3 = nn.Linear(512, 256)
        # nn.init.normal_(self.layer_3.weight, 0., 0.3)
        # nn.init.constant_(self.layer_3.bias, 0.1_episodes2000_steps99_var3)
        self.ln3 = nn.LayerNorm(256)

        self.layer_4 = nn.Linear(256, 128)
        # nn.init.normal_(self.layer_3.weight, 0., 0.3)
        # nn.init.constant_(self.layer_3.bias, 0.1_episodes2000_steps99_var3)
        self.ln4 = nn.LayerNorm(128)

        self.output = nn.Linear(128, action_dim)
        # self.output.weight.data.normal_(0., 0.3)
        # self.output.bias.data.fill_(0.1_episodes2000_steps99_var3)

    def forward(self, s):
        a = F.relu(self.ln1(self.layer_1(s)))
        a = F.relu(self.ln2(self.layer_2(a)))
        a = F.relu(self.ln3(self.layer_3(a)))
        a = F.relu(self.ln4(self.layer_4(a)))
        a = torch.tanh(self.output(a))
        # 对action进行放缩，实际上a in [-1_episodes2000_steps99_var3,1_episodes2000_steps99_var3]
        # a = self.bn(a)
        # scaled_a = a * self.action_bound
        # scaled_a = a
        # return scaled_a
        return a


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 1024
        # layer
        self.layer_1 = nn.Linear(state_dim + action_dim, n_layer)
        # nn.init.normal_(self.layer_1.weight, 0., 0.1_episodes2000_steps99_var3)
        # nn.init.constant_(self.layer_1.bias, 0.1_episodes2000_steps99_var3)
        self.ln1 = nn.LayerNorm(1024)

        self.layer_2 = nn.Linear(1024, 512)
        # nn.init.normal_(self.layer_2.weight, 0., 0.1_episodes2000_steps99_var3)
        # nn.init.constant_(self.layer_2.bias, 0.1_episodes2000_steps99_var3)
        self.ln2 = nn.LayerNorm(512)

        self.layer_3 = nn.Linear(512, 258)
        # nn.init.normal_(self.layer_3.weight, 0., 0.1_episodes2000_steps99_var3)
        # nn.init.constant_(self.layer_3.bias, 0.1_episodes2000_steps99_var3)
        self.ln3 = nn.LayerNorm(258)

        self.layer_4 = nn.Linear(258, 128)
        # nn.init.normal_(self.layer_4.weight, 0., 0.1_episodes2000_steps99_var3)
        # nn.init.constant_(self.layer_4.bias, 0.1_episodes2000_steps99_var3)
        self.ln4 = nn.LayerNorm(128)

        self.output = nn.Linear(128, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], 2)
        x = self.layer_1(x)
        x = self.layer_2(F.relu(self.ln1(x)))
        x = self.layer_3(F.relu(self.ln2(x)))
        x = self.layer_4(F.relu(self.ln3(x)))
        # x = self.layer_4(F.relu(x))
        # a = self.layer_2(a)
        q_val = self.output(F.relu(self.ln4(x)))
        return q_val


# Deep Deterministic Policy Gradient

class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, replacement=1, gamma=0.9, lr_a=0.000001,
                 lr_c=0.0001, batch_size=32):
        super(DDPG, self).__init__()

        # self.resnet18 = torchvision.models.resnet18()
        # self.resnet18.fc = nn.Sequential()
        # self.resnet18.to(device)
        # self.state_dim = 512

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size

        # 记忆库
        self.pointer = 0
        # 定义 Actor 网络
        self.actor = Actor(self.state_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(self.state_dim, action_dim, action_bound).to(device)
        # 定义 Critic 网络
        self.critic = Critic(self.state_dim, action_dim).to(device)
        self.critic_target = Critic(self.state_dim, action_dim).to(device)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss().to(device)




    def act(self, s):
        s = torch.FloatTensor(s).to(device)
        action = self.actor(s)
        return action.detach().cpu().numpy()

    def choose_action(self, s):
        #with resnet
        # s = torch.FloatTensor(s).unsqueeze(0).to(device)
        # s = self.resnet18(s)
        # action = self.actor(s)
        # return action.detach().cpu().squeeze(0).numpy()

        s = torch.FloatTensor(s).to(device)
        action = self.actor(s)
        return action.detach().cpu().numpy()

    def learn(self, memory, batchsize):

        # soft replacement and hard replacement
        # 用于更新target网络的参数
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            for al in a_layers:
                a = self.actor.state_dict()[al[0] + '.weight']
                al[1].weight.data.mul_((1 - tau)) #mul_表示乘起来再赋给原来的值，add_同理
                al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])
                al[1].bias.data.mul_((1 - tau))
                al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])
            for cl in c_layers:
                cl[1].weight.data.mul_((1 - tau))
                cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])
                cl[1].bias.data.mul_((1 - tau))
                cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])

        else:
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']

            self.t_replace_counter += 1

        # 从记忆库中采样bacth data
        bs, ba, br, bs_, bdone = memory.sample(batchsize)
        bs = torch.FloatTensor(np.float32(bs)).to(device)
        ba = torch.FloatTensor(np.float32(ba)).to(device)
        br = torch.FloatTensor(np.float32(br)).to(device)
        bs_ = torch.FloatTensor(np.float32(bs_)).to(device)

        # bm = self.sample()
        # bs = torch.FloatTensor(bm[:, :self.state_dim]).to(device)
        # ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim]).to(device)
        # br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim]).to(device)
        # bs_ = torch.FloatTensor(bm[:, -self.state_dim:]).to(device)

        # 训练Actor
        a = self.actor(bs)
        q = self.critic(bs, a)
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        #a_loss.backward(retain_graph=True)
        a_loss.backward()
        self.aopt.step()

        # 训练critic
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br.view(-1,1,1) + self.gamma * self.critic_target(bs_, a_)
        q_eval =                     self.critic(bs, ba)
        td_error = self.mse_loss(q_target, q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

        return td_error.cpu(), a_loss.cpu()
        # if point%100==0:
        #     print(-a_loss,td_error)


    def save(self, path):
        ckpt = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_bound': self.action_bound,
            'model': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
        torch.save(ckpt, path)
        del ckpt