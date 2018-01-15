# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gym

# 定义一些超参数


batch_size = 32
lr = 0.01
epsilon = 0.9
gamma = 0.9
target_replace_iter = 100
memory_capacity = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]


class q_net(nn.Module):
    def __init__(self, hidden=50):
        super(q_net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, n_actions)
        )

        nn.init.normal(self.fc[0].weight, std=0.1)  # 使用标准差是 0.1 的正态分布初始化
        nn.init.normal(self.fc[2].weight, std=0.1)  # 使用标准差是 0.1 的正态分布初始化

    def forward(self, x):
        actions_value = self.fc(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = q_net(), q_net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))  # 当前的状态和动作，之后的状态和动作
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def choose_action(self, s):
        '''
        根据输入的状态得到所有可行动作的价值估计
        '''
        s = Variable(torch.unsqueeze(torch.FloatTensor(s), 0))
        # input only one sample
        if np.random.uniform() < epsilon:  # greedy 贪婪算法
            actions_value = self.eval_net(s)
            action = torch.max(actions_value, 1)[1].data[0]
        else:  # random 随机选择
            action = np.random.randint(0, n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 用新的记忆替换旧的记忆
        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 的参数更新
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 取样记忆中的经历
        sample_index = np.random.choice(memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :n_states]))
        b_a = Variable(
            torch.LongTensor(b_memory[:, n_states:n_states + 1].astype(int)))
        b_r = Variable(
            torch.FloatTensor(b_memory[:, n_states + 1:n_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -n_states:]))

        # q_eval net 评估状态下动作的 value
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 选择对应 action 的动作
        q_next = self.target_net(
            b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)  # shape (batch, 1)
        loss = self.criterion(q_eval, q_target)  # mse 作为 loss 函数
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn_trainer = DQN()

print('collecting experience ... ')
all_reward = []
for i_episode in range(300):
    s = env.reset()
    reward = 0
    while True:
        if dqn_trainer.memory_counter > memory_capacity:
            env.render()
        a = dqn_trainer.choose_action(s)

        # 环境采取动作得到结果
        s_, r, done, info = env.step(a)

        # 修改奖励以便更快收敛
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn_trainer.store_transition(s, a, r, s_)

        reward += r
        if dqn_trainer.memory_counter > memory_capacity:  # 记忆收集够开始学习
            dqn_trainer.learn()
            if done:
                print('Ep: {} | reward: {:.3f}'.format(i_episode, round(reward, 3)))
                all_reward.append(reward)
                break

        if done:
            break
        s = s_
