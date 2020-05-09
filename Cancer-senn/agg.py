
import json
import numpy as np
import pandas as pd

with open("trial1/out.json") as f:
    data = json.load(f)

dataset = "cancer"
columns = list(data.keys())

df = pd.DataFrame(0, index = [dataset], columns = columns)

n_trials = 10
for i in range(n_trials):
    
    with open("trial" + str(i) + "/out.json") as f:
        data = json.load(f)
        
    for name in columns:
        df.loc[dataset, name] += data[name] / n_trials
    
    
df.to_csv("results_mean.csv")


df_sd = pd.DataFrame(0, index = [dataset], columns = columns)
for i in range(n_trials):
    
    with open("trial" + str(i) + "/out.json") as f:
        data = json.load(f)
        
        for name in columns:
            delta = data[name] - df.loc[dataset, name]
            df_sd.loc[dataset, name] += delta**2 / (n_trials - 1)
                
for name in columns:
    df_sd.loc[dataset, name] = np.sqrt(df_sd.loc[dataset, name])

df_sd.to_csv("results_sd.csv")
