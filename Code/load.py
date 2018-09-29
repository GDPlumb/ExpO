
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_normalize_data(source, thresh = .0000000001):
    df_train = pd.read_csv(source, header = None).dropna()
    
    # Split train, test, valid - Change up train valid test every iteration
    df_train, df_test = train_test_split(df_train, test_size = 0.5)
    df_valid, df_test = train_test_split(df_test, test_size = 0.5)
    
    # delete features for which all entries are equal (or below a given threshold)
    train_stddev = df_train[df_train.columns[:]].std()
    drop_small = np.where(train_stddev < thresh)
    if train_stddev[df_train.shape[1] - 1] < thresh:
        print("ERROR: Near constant predicted value")
    df_train = df_train.drop(drop_small[0], axis = 1)
    df_test = df_test.drop(drop_small[0], axis = 1)
    df_valid = df_valid.drop(drop_small[0], axis = 1)
    
    # Calculate std dev and mean
    train_stddev = df_train[df_train.columns[:]].std()
    train_mean = df_train[df_train.columns[:]].mean()
    
    # Normalize to have mean 0 and variance 1
    df_train1 = (df_train - train_mean) / train_stddev
    df_valid1 = (df_valid - train_mean) / train_stddev
    df_test1 = (df_test - train_mean) / train_stddev
    
    # Convert to np arrays
    X_train = df_train1[df_train1.columns[:-1]].values
    y_train = df_train1[df_train1.columns[-1]].values
    
    X_valid = df_valid1[df_valid1.columns[:-1]].values
    y_valid = df_valid1[df_valid1.columns[-1]].values
    
    X_test = df_test1[df_test1.columns[:-1]].values
    y_test = df_test1[df_test1.columns[-1]].values
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, np.array(train_mean), np.array(train_stddev)

