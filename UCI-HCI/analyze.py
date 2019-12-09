
import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv("results.csv", header = None).values

none = np.where(data[:, 0] == 0)[0]
none_vals = data[none, 1]

expo = np.where(data[:, 0] == 1)[0]
expo_vals = data[expo, 1]

print("Mean None: ", np.mean(none_vals))
print("Mean ExpO: ", np.mean(expo_vals))
print(stats.ttest_ind(none_vals,expo_vals))
