
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Data import DataManager, BatchManager

class CancerDataManager(DataManager):

    def __init__(self, source, train_batch_size = 20, reg_batch_size = None):
        self.source = source
        self.train_batch_size = train_batch_size
        self.reg_batch_size = reg_batch_size
        
        data = pd.read_csv(source)

        # Format the Labels
        y = data.diagnosis
        y = 1.0 * (y.values == 'M')
        y = np.expand_dims(y, 1)

        # Format the Features
        list = ['Unnamed: 32','id','diagnosis']
        x = data.drop(list,axis = 1 )
        x = x.values
        mu = np.mean(x, axis = 0)
        sd = np.std(x, axis = 0)
        x = (x - mu) / sd
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1)
    
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.train_batch_manager = BatchManager(X_train, y_train)
    
        if reg_batch_size != None:
            self.reg_batch_manager = BatchManager(X_train, y_train)




