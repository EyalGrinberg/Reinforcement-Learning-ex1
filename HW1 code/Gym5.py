import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
a = []
s = 0
for j in range(1000):
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
    if total_reward == 200:
        a.append(i+1)
        s += i+1
plt.hist(a, bins = 30, rwidth=0.5)
plt.title("Distribution of Required Number of Episodes Until Optimum")
plt.xlabel("Number of Episodes")
plt.ylabel("Frequency")
plt.show()
print(s / 1000)
