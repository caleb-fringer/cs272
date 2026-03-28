import myagent
import runner
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

filepath = sys.argv[1]
with open(filepath, "rb") as f:
    q_values = pickle.load(f)

i,j = [max(dim) for dim in zip(*q_values)]
v_values = np.full((i+1,j+1), -400)
for (i,j), actions in q_values.items():
    optimal_action_value = max(actions.values())
    v_value = 0.9*optimal_action_value
    for q_value in actions.values():
        v_value += (1/40) * q_value
    v_values[i][j] = v_value

annot_matrix = np.where(v_values <= -400, "", np.round(v_values, 1).astype(str))
sns.heatmap(v_values, annot=False, cmap="rocket")
plt.show()
