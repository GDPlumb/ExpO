import json
import math
import pandas as pd
from scipy import stats

###
# Configure what results to compare
###

dir1 = "../1-Initial/Trials/"
dir2 = "../2-Causal/Trials/"

datasets = ["autompgs", "communities", "crimes", "day", "happiness", "housing", "music", "winequality-red"]
trials = []
for i in range(10):
    trials.append(i + 1)

###
# Stat Testing
###

with open(dir1 + datasets[0] + "_" + str(trials[0]) + ".json") as f:
    data = json.load(f)
columns = list(data.keys())

df1 = pd.DataFrame(index = datasets, columns = columns)
df1 = df1.apply(lambda x:x.apply(lambda x:[] if math.isnan(x) else x))

df2 = pd.DataFrame(index = datasets, columns = columns)
df2 = df2.apply(lambda x:x.apply(lambda x:[] if math.isnan(x) else x))

out = pd.DataFrame(index = datasets, columns = ["Acc", "Standard", "Causal", "Stability"])

for dataset in datasets:
    for trial in trials:
    
        with open(dir1 + dataset + "_" + str(trial) + ".json") as f:
            data = json.load(f)
        for name in columns:
            df1.ix[dataset, name].append(data[name])

        with open(dir2 + dataset + "_" + str(trial) + ".json") as f:
            data = json.load(f)
        for name in columns:
            df2.ix[dataset, name].append(data[name])

    out.ix[dataset, "Acc"] = stats.ttest_ind(df1.ix[dataset, "test_acc"],df2.ix[dataset, "test_acc"], equal_var = False).pvalue
    out.ix[dataset, "Standard"] = stats.ttest_ind(df1.ix[dataset, "standard_metric"],df2.ix[dataset, "standard_metric"], equal_var = False).pvalue
    out.ix[dataset, "Causal"] = stats.ttest_ind(df1.ix[dataset, "causal_metric"],df2.ix[dataset, "causal_metric"], equal_var = False).pvalue
    out.ix[dataset, "Stability"] = stats.ttest_ind(df1.ix[dataset, "stability_metric"],df2.ix[dataset, "stability_metric"], equal_var = False).pvalue
    
out.to_csv("results.csv")
