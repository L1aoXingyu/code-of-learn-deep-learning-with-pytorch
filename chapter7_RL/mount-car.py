import numpy as np

import gym

n_states = 40  # 取样 40 个状态
iter_max = 10000

initial_lr = 1.0  # Learning rate
min_lr = 0.003
gamma = 1.0
t_max = 10000
eps = 0.02


def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:  # 如果没有策略，就随机取样
            action = env.action_space.sample()
        else:
            a, b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward


def obs_to_state(env, obs):
    """
    将观察的连续环境映射到离散的输入的状态
    """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    return a, b


if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    print('----- using Q Learning -----')
    q_table = np.zeros((n_states, n_states, 3))
    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0
        ## eta: 每一步学习率都不断减小
        eta = max(min_lr, initial_lr * (0.85 ** (i // 100)))
        for j in range(t_max):
            x, y = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < eps:  # greedy 贪心算法
                action = np.random.choice(env.action_space.n)
            else:
                logits = q_table[x, y, :]
                logits_exp = np.exp(logits)
                probs = logits_exp / np.sum(logits_exp)  # 算出三个动作的概率
                action = np.random.choice(env.action_space.n, p=probs)  # 依概率来选择动作
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            # 更新 q 表
            x_, y_ = obs_to_state(env, obs)
            q_table[x, y, action] = q_table[x, y, action] + eta * (
                    reward + gamma * np.max(q_table[x_, y_, :]) -
                    q_table[x, y, action])
            if done:
                break
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' % (i + 1,
                                                           total_reward))
    solution_policy = np.argmax(q_table, axis=2)  # 在 q 表中每个状态下都取最大的值得动作
    solution_policy_scores = [
        run_episode(env, solution_policy, False) for _ in range(100)
    ]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    run_episode(env, solution_policy, True)
