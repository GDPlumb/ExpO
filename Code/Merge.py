
import json
import numpy as np
import pandas as pd

def merge(datasets, trials):

    with open("Trials/" + datasets[0] + "_" + str(trials[0]) + ".json") as f:
        data = json.load(f)

    columns = list(data.keys())
    df = pd.DataFrame(0, index = datasets, columns = columns)
    df = df.astype("object")

    for dataset in datasets:
        for trial in trials:
            with open("Trials/" + dataset + "_" + str(trial) + ".json") as f:
                data = json.load(f)
            for name in columns:
                df.ix[dataset, name] += np.asarray(data[name]) / len(trials)

    df.to_csv("results.csv")
