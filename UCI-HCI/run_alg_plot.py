
import matplotlib.pyplot as plt
import numpy as np

out = np.loadtxt("agent.csv", delimiter=',')

plt.plot(out[:, 0], out[:, 1], label = "Agent - None")
plt.plot(out[:, 0], out[:, 2], label = "Agent - ExpO")
plt.plot(out[:, 0], [11.45] * out.shape[0], "C0", label = "Human - None", linestyle = "--")
plt.plot(out[:, 0], [8.0] * out.shape[0], "C1", label = "Human - ExpO", linestyle = "--")
plt.ylabel("Average Number of Steps")
plt.xlabel("Agent Score Scale Factor")
plt.legend()
plt.savefig("agent.pdf")
plt.close()
