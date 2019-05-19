"""Data manager for Million Song dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from collections import defaultdict

import numpy as np

from sklearn.model_selection import train_test_split

from Data import DataManager

N_DIMS = 90
MAX_PER_LABEL = 500
YEAR_RANGE = (1920, 2014)


def parse_point(point, n_dims=N_DIMS):
    point_data = list(map(float, point.split(',')))
    label, features = int(point_data[0]), np.asarray(point_data[1:])
    assert len(features) == n_dims, "Unexpected number of features!"
    return features, label


def parse_file(
    filename,
    n_points,
    n_dims=N_DIMS,
    max_per_label=MAX_PER_LABEL,
    year_range=YEAR_RANGE,
):
    X = np.empty([n_points, n_dims])
    y = np.empty(n_points)
    count = 0

    year_dict = defaultdict(int)
    with open(filename, "r") as fp:
        for line in fp:
            features, label = parse_point(line, n_dims)
            if year_dict[label] < max_per_label and label > year_range[0]:
                y[count] = label
                X[count] = features
                year_dict[label] += 1
                count += 1

    return X[:count, :], y[:count]


def scale_data_alt(X, X_mean=None, X_max=None, X_min=None, eps=1e-10):
    # de-mean
    if X_mean is None:
        X_mean = np.mean(X, axis=0, keepdims=True)
    X = X - X_mean

    # make all features 0-1 (but don't control variance)
    if X_max is None:
        X_max = np.max(X, axis=0, keepdims=True)
    if X_min is None:
        X_min = np.min(X, axis=0, keepdims=True)
    X = (X - X_min) / (X_max - X_min + eps)

    return X, (X_mean, X_min, X_max)


class YearPredictionMSDDataManager(DataManager):
    """Data manager for year prediction in the million song dataset.

    Note: We treat the year as a regression target.
    """
    N_TRAIN = 463715
    N_TEST = 51630
    
    def load_normalize_data(self, valid_size=0.1, thresh=1e-10):
        train_path = os.path.join(self.source, "YearPredictionMSD_train.txt")
        test_path = os.path.join(self.source, "YearPredictionMSD_test.txt")

        # Load data.
        X_train, y_train = parse_file(train_path, self.N_TRAIN)
        X_test, y_test = parse_file(test_path, self.N_TEST)
        
        y_train = np.expand_dims(y_train, 1)
        y_test = np.expand_dims(y_test, 1)
        
        # Split train into train and validation.
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = valid_size)
        
        # Normalize data.
        train_mean = np.mean(X_train, axis = 0)
        train_std = np.std(X_train, axis = 0)
        
        X_train = (X_train - train_mean) / train_std
        X_valid = (X_valid - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std
        
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        
        y_train = (y_train - y_mean) / y_std
        y_valid = (y_valid - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        return X_train, y_train, X_valid, y_valid, X_test, y_test, np.append(train_mean, y_mean), np.append(train_std, y_std)

    def load_normalize_data_alt(self, valid_size=0.1, thresh=1e-10, seed=0):
        train_path = os.path.join(self.source, "YearPredictionMSD_train.txt")
        test_path = os.path.join(self.source, "YearPredictionMSD_test.txt")

        # Load data.
        X_train, y_train = parse_file(train_path, self.N_TRAIN)
        X_test, y_test = parse_file(test_path, self.N_TEST)
        
        y_train = np.expand_dims(y_train, 1)
        y_test = np.expand_dims(y_test, 1)
        
        # Normalize data.
        X_train, (X_mean, X_min, X_max) = scale_data_alt(X_train)
        X_test, _ = scale_data_alt(X_test, X_mean, X_max, X_min)
        y_test -= np.min(y_train)
        y_train -= np.min(y_train)

        # Split train into train and validation.
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=valid_size, random_state=seed
        )

        return (
            X_train, y_train,
            X_valid, y_valid,
            X_test, y_test,
            X_mean, (X_max - X_min)
        )
