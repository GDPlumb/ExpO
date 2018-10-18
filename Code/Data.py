
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager():

    def __init__(self, source, train_batch_size = 20, reg_batch_size = None):
        self.source = source
        self.train_batch_size = train_batch_size
        self.reg_batch_size = reg_batch_size
    
        X_train, y_train, X_val, y_val, X_test, y_test, mu, sigma = self.load_normalize_data()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.mu = mu
        self.sigma = sigma
    
        self.train_batch_manager = BatchManager(X_train, y_train)
    
        if reg_batch_size != None:
            self.reg_batch_manager = BatchManager(X_train, y_train)

    def link_graph(self, X, Y, X_reg = None):
        self.X = X
        self.Y = Y
        if X_reg != None:
            self.X_reg = X_reg

    def train_feed(self):
        batch_x, batch_y = self.train_batch_manager.next_batch(self.train_batch_size)
        if self.reg_batch_size == None:
            return {self.X: batch_x, self.Y: batch_y}
        else:
            batch_x_reg, batch_y_reg = self.reg_batch_manager.next_batch(self.reg_batch_size)
            return {self.X: batch_x, self.Y: batch_y, self.X_reg: batch_x_reg}

    def eval_feed(self, val = False, scale = 1.0):
        if val:
            X_eval = self.X_val
            y_eval = self.y_val
        else:
            X_eval = self.X_train
            y_eval = self.y_train
        indices = np.random.choice(X_eval.shape[0], np.int(scale * self.train_batch_size), replace = False)
        batch_x = X_eval[indices]
        batch_y = y_eval[indices]
        if self.reg_batch_size == None:
            return {self.X: batch_x, self.Y: batch_y}
        else:
            indices_reg = np.random.choice(X_eval.shape[0], np.int(scale * self.reg_batch_size), replace = False)
            batch_x_reg = X_eval[indices_reg]
            return {self.X: batch_x, self.Y: batch_y, self.X_reg: batch_x_reg}

    def load_data(self):
        return pd.read_csv(self.source, header = None).dropna()

    def load_normalize_data(self, thresh = .0000000001):
        df_train = self.load_data()
        
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


# Source: https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
# Modification: added a response variable to the dataset
class BatchManager():

    def __init__(self, X, Y):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._X = X
        self._num_examples = X.shape[0]
        self._Y = Y

    @property
    def X(self):
        return self._X
    
    @property
    def Y(self):
        return self._Y

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        
        # Shuffle the data on the first call
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self._X = self.X[idx]
            self._Y = self.Y[idx]
        
        # If there aren't enough points left in this epoch to fill the minibatch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start

            # Load the remaining data
            X_rest_part = self.X[start:self._num_examples]
            Y_rest_part = self.Y[start:self._num_examples]
            
            # Reshuffle the dataset
            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            self._X = self.X[idx0]
            self._Y = self.Y[idx0]
            
            # Get the remaining samples for the batch from the next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end =  self._index_in_epoch
            X_new_part = self._X[start:end]
            Y_new_part = self._Y[start:end]
            return np.concatenate((X_rest_part, X_new_part), axis=0), np.concatenate((Y_rest_part, Y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end], self._Y[start:end]
