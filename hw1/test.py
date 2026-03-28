import matplotlib.pyplot as plt
import numpy as np

a = 1.0
decay_rate = 1/4
base = 0.99

result = [1.0]
for i in range(1000):
    a = a * base**decay_rate
    result.append(a) 

result2 =np.array(list(map(lambda x: 1.0*0.99**(x), np.arange(1000))))
plt.plot(result)
plt.plot(result2)
plt.show()

