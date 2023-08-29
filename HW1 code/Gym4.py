import gym
import numpy as np

env = gym.make('CartPole-v0')
for i in range(10000):
    w = np.random.uniform(-1, 1, 4)
    obs, info = env.reset()
    total_reward = 0
    while (True):
        action = 1 if (np.matmul(w, obs) >= 0) else 0
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    if total_reward == 200:
        break
print(i)
print(w)