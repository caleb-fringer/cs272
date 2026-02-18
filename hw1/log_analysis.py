import re
import matplotlib.pyplot as plt
def parse_reward(line):
    rewards = re.findall(r"(?:Total Reward: )(-?\d+\.\d+)",line)
    if len(rewards) != 1:
        raise ValueError("Expected 1 reward per line")
    return float(rewards[0])

with open("logs/run_2026-02-17_14-38-00.log") as file:
    returns = [parse_reward(line) for line in file.readlines() if line.find("Total Reward: ") > 0 ]

returns
len(returns)
import numpy as np
np.average(returns[-20:])

count = 0
i = 0
while count < 5:
    if returns[i] > -400.0:
        count += 1
    i += 1
i
n = len(returns) - i + 1
((0.99)**n)*0.5

def simple_moving_avg(data, window_size=20):
    weights = np.ones(window_size) / window_size
    sma = np.convolve(data, weights, mode="valid")
    return sma

max(returns)
sma = simple_moving_avg(returns)
sma
plt.plot(simple_moving_avg(returns))
plt.show()
