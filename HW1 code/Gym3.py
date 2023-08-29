import gym
import numpy as np

env = gym.make('CartPole-v0')

w = np.random.uniform(-1, 1, 4)
total_reward = 0
obs, info = env.reset()
while (True):
    action = 1 if (np.matmul(w, obs) >= 0) else 0
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
print(total_reward)