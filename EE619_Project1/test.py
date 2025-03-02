import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# Q-learning 알고리즘
def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=10000):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # epsilon-greedy 정책에 따른 행동 선택
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Q함수 업데이트
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            state = next_state

        rewards.append(episode_reward)

    return rewards

# Double Q-learning 알고리즘
def double_q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=10000):
    q_table1 = np.zeros((env.observation_space.n, env.action_space.n))
    q_table2 = np.zeros((env.observation_space.n, env.action_space.n))

    rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # epsilon-greedy 정책에 따른 행동 선택
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table1[state] + q_table2[state])

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # 두 개의 Q함수 중 작은 값을 사용하여 업데이트
            if np.random.uniform() < 0.5:
                q_table1[state, action] += alpha * (reward + gamma * q_table2[next_state, np.argmax(q_table1[next_state])] - q_table1[state, action])
            else:
                q_table2[state, action] += alpha * (reward + gamma * q_table1[next_state, np.argmax(q_table2[next_state])] - q_table2[state, action])

            state = next_state

        rewards.append(episode_reward)

    return rewards

# 각 알고리즘 실행 및 결과 비교
q_learning_rewards = q_learning(env)
double_q_learning_rewards = double_q_learning(env)

plt.plot(q_learning_rewards, label='Q-learning')
plt.plot(double_q_learning_rewards, label='Double Q-learning')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.legend()
plt.show()