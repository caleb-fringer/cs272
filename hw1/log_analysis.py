import re
import matplotlib.pyplot as plt
import sys

path = sys.argv[1]
def parse_reward(line):
    rewards = re.findall(r"(?:Total Reward: )(-?\d+\.\d+)",line)
    if len(rewards) != 1:
        raise ValueError("Expected 1 reward per line")
    return float(rewards[0])

with open(path) as file:
    returns = [parse_reward(line) for line in file.readlines() if line.find("Total Reward: ") > 0 ]

import numpy as np
np.average(returns[-20:])

def simple_moving_avg(data, window_size=20):
    weights = np.ones(window_size) / window_size
    sma = np.convolve(data, weights, mode="valid")
    return sma

plt.plot(simple_moving_avg(returns))
plt.title("Avg. Return by Episode")
plt.xlabel("Episodes")
plt.ylabel("Avg. Return (past 20 episodes)")
plt.show()
