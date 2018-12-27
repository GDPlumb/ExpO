"""Data managers for binary classification datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Data import DataManager

class BinaryClassificationDataManager(DataManager):
    """Data manager that doesn't try to normalize targets."""

    def load_normalize_data(self, thresh=1e-10):
        df_train = self.load_data()

        # Split train, test, valid - Change up train valid test every iteration
        df_train, df_test = train_test_split(df_train, test_size=0.8)
        df_valid, df_test = train_test_split(df_test, test_size=0.5)

        # delete features for which all entries are equal (or below a given threshold)
        train_stddev = df_train[df_train.columns[:-1]].std()
        drop_small = df_train.columns[np.where(train_stddev < thresh)[0]]
        df_train = df_train.drop(drop_small, axis=1)
        df_test = df_test.drop(drop_small, axis=1)
        df_valid = df_valid.drop(drop_small, axis=1)

        # Calculate std dev and mean
        X_train = df_train[df_train.columns[:-1]]
        X_valid = df_valid[df_valid.columns[:-1]]
        X_test = df_test[df_test.columns[:-1]]

        X_train_stddev = X_train.std()
        X_train_mean = X_train.mean()

        # Normalize to have mean 0 and variance 1
        #X_train = (X_train - X_train_mean) / X_train_stddev
        #X_valid = (X_valid - X_train_mean) / X_train_stddev
        #X_test = (X_test - X_train_mean) / X_train_stddev

        # Convert to np arrays
        X_train = X_train.values
        y_train = np.expand_dims(df_train[df_train.columns[-1]].values, axis = 1)

        X_valid = X_valid.values
        y_valid = np.expand_dims(df_valid[df_valid.columns[-1]].values, axis = 1)

        X_test = X_test.values
        y_test = np.expand_dims(df_test[df_test.columns[-1]].values, axis = 1)

        return (X_train, y_train, X_valid, y_valid, X_test, y_test,
                np.array(X_train_mean), np.array(X_train_stddev))
