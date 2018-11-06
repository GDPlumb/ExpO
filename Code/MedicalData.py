"""Data managers for medical datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Data import DataManager

class ClassificationDataManager(DataManager):
    """Data manager that doesn't try to normalize targets."""

    def load_normalize_data(self, thresh=1e-10):
        df_train = self.load_data()

        # Split train, test, valid - Change up train valid test every iteration
        df_train, df_test = train_test_split(df_train, test_size=0.5)
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
        X_train = (X_train - X_train_mean) / X_train_stddev
        X_valid = (X_valid - X_train_mean) / X_train_stddev
        X_test = (X_test - X_train_mean) / X_train_stddev

        # Convert to np arrays
        X_train = X_train.values
        y_train_idx = df_train[df_train.columns[-1]].values
        # idx -> one-hot
        y_train = np.zeros((X_train.shape[0], 3))
        y_train[np.arange(y_train.shape[0]), y_train_idx] = 1

        X_valid = X_valid.values
        y_valid_idx = df_valid[df_valid.columns[-1]].values
        # idx -> one-hot
        y_valid = np.zeros((X_valid.shape[0], 3))
        y_valid[np.arange(y_valid.shape[0]), y_valid_idx] = 1

        X_test = X_test.values
        y_test_idx = df_test[df_test.columns[-1]].values
        # idx -> one-hot
        y_test = np.zeros((X_test.shape[0], 3))
        y_test[np.arange(y_test.shape[0]), y_test_idx] = 1

        return (X_train, y_train, X_valid, y_valid, X_test, y_test,
                np.array(X_train_mean), np.array(X_train_stddev))


class HospitalReadmissionDataManager(ClassificationDataManager):
    """Data manager for the hospital readmission data."""

    # Data parameters.
    # _ECLUDE_COLUMNS = []
    _EXCLUDE_COLUMNS = ["diag_1", "diag_2", "diag_3"]

    def load_data(self):
        df = pd.read_csv(self.source, na_values="?")

        # Exclude certain columns.
        df.drop(self._EXCLUDE_COLUMNS, axis=1)

        # Fill N/A.
        df.fillna({"race": "Other"})

        # Convert categorical columns to one-hot.
        df_onehot = pd.get_dummies(df[df.columns[:-1]], dummy_na=True)
        df_onehot["readmitted"] = pd.Categorical(df[df.columns[-1]]).codes

        return df_onehot


class Support2DataManager(ClassificationDataManager):
    """Data manager for the SUPPORT2 dataset."""

    # Data parameters.
    _TRAIN_SIZE, _VALID_SIZE, _TEST_SIZE = 7105, 1000, 1000
    _EXCLUDE_FEATURES = [
        "aps", "sps", "surv2m", "surv6m", "prg2m", "prg6m", "dnr", "dnrday",
        "hospdead", "dzclass", "edu", "scoma", "totmcst", "charges", "totcst",
    ]
    _TARGETS = ["death", "d.time"]
    _AVG_VALUES = {
        "alb": 3.5,
        "bili": 1.01,
        "bun": 6.51,
        "crea": 1.01,
        "pafi": 333.3,
        "wblc": 9.,
        "urine": 2502.,
    }

    def load_data(self, fill_na="avg", na_value=-1.0):
        df = pd.read_csv(self.source)
        columns = sorted(list(set(df.columns) - set(self._EXCLUDE_FEATURES)))
        df = df[columns]

        # Split into features and targets.
        targets = df[self._TARGETS]
        features = df[list(set(df.columns) - set(self._TARGETS))]

        # Convert categorical columns into one-hot format
        cat_columns = features.columns[features.dtypes == "object"]
        features = pd.get_dummies(features, dummy_na=False, columns=cat_columns)

        # Scale and impute real-valued features.
        features[["num.co", "slos", "hday"]] = \
            features[["num.co", "slos", "hday"]].astype(np.float)
        float_cols = features.columns[features.dtypes == np.float]
        features[float_cols] = \
            (features[float_cols] - features[float_cols].min()) / \
            (features[float_cols].max() - features[float_cols].min())
        if fill_na == "avg":
            for key, val in self._AVG_VALUES.items():
                features[[key]] = features[[key]].fillna(val)
        features.fillna(na_value, inplace=True)

        # Construct final data frame.
        df = features
        df["death"] = pd.Categorical(targets["death"]).codes

        return df
