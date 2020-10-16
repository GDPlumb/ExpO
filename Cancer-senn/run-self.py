# -*- coding: utf-8 -*-
""" Code for training and evaluating Self-Explaining Neural Networks.
Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from scipy.special import expit as sigmoid

# Standard Imports
import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator
import matplotlib
matplotlib.use('Agg')
import json


import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd

# Torch Imports
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader

# Scikit-learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local imports
from os.path import dirname, realpath
sys.path.append(os.path.join(dirname(realpath(__file__)),'codebase/'))


from SENN.arglist import parse_args
from SENN.utils import plot_theta_stability, generate_dir_names
from SENN.eval_utils import sample_local_lipschitz, estimate_dataset_lipschitz
from SENN.models import GSENN
from SENN.conceptizers import input_conceptizer
from SENN.parametrizers import  dfc_parametrizer
from SENN.aggregators import additive_scalar_aggregator
from SENN.trainers import VanillaClassTrainer, GradPenaltyTrainer, CLPenaltyTrainer

def load_cancer_data(valid_size=0.1, shuffle=True, batch_size=64):
    data = pd.read_csv("../../Datasets/cancer.csv")
    x = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
    diag = { "M": 1, "B": 0}
    y = data["diagnosis"].replace(diag)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5)
    Tds = []
    Loaders = []
    Data = []
    for (foldx, foldy) in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        scaler = StandardScaler()
        transformed = scaler.fit_transform(foldx)
        tds = TensorDataset(torch.from_numpy(transformed).float(),
                            torch.from_numpy(foldy.as_matrix()).view(-1, 1).float())
        loader = DataLoader(tds, batch_size=batch_size, shuffle=False)
        Tds.append(tds)
        Loaders.append(loader)
        Data.append(transformed)
    return (*Loaders, *Tds, data, *Data)
    
class Wrapper():
    def __init__(self, model):
        self.model = model
        
    def predict(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        x_torch = Variable(torch.from_numpy(x)).float()
        return self.model(x_torch).cpu().data.numpy()
        
    def explain(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        x_torch = Variable(torch.from_numpy(x)).float()
        return self.model.parametrizer(x_torch).cpu().data.numpy()
        
    def concept(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        x_torch = Variable(torch.from_numpy(x)).float()
        return self.model.conceptizer(x_torch).cpu().data.numpy()
        
def pred_from_coefs(coefs, x):
    return sigmoid(np.sum(coefs * np.append(x, 1)))
    
def generate_neighbor(x, stddev = 0.1):
    return x + stddev * np.random.normal(loc=0.0, scale=1.0, size = x.shape)

    
# Evaluate MAPLE as a black-box explainer for this model
def metrics_self(model, X_test, stddev = 0.1):

    num_perturbations = 5

    # Get the model predictions on data
    test_pred = model.predict(X_test)
    
    # Get the necessary sizes
    n_test = X_test.shape[0]
    d_in = X_test.shape[1]

    # Compute the standard, causal, and stability metrics
    standard_metric = 0.0
    causal_metric = 0.0
    stability_metric = 0.0
    
    for i in range(n_test):

        x = X_test[i, :]
        

        # Get SENN's Explanation
        e = model.explain(x)

        # Standard Metric
        standard_metric += (pred_from_coefs(e, x) - test_pred[i])**2

        for k in range(num_perturbations):
            x_pert = generate_neighbor(x, stddev = stddev)

            # Causal Metric
            model_pred = model.predict(x_pert.reshape(1, d_in))
            exp_pred = pred_from_coefs(e, x_pert)
            causal_metric += (exp_pred - model_pred)**2

            # Stability Metric
            e_pert = model.explain(x_pert)
            stability_metric += np.sum((e_pert - e)**2)

    standard_metric /= n_test
    causal_metric /= num_perturbations * n_test
    stability_metric /= num_perturbations * n_test

    return standard_metric, causal_metric, stability_metric
            
def main():
    args = parse_args()
    args.nclasses = 1
    args.theta_dim = args.nclasses
    args.print_freq = 100
    args.epochs = 10
    train_loader, valid_loader, test_loader, train, valid, test, data, x_train, x_valid, x_test  = load_cancer_data()
    
    layer_sizes = (10,10,5)
    input_dim = 30

    # model

    if args.h_type == 'input':
        conceptizer  = input_conceptizer()
        args.nconcepts = input_dim + int(not args.nobias)
    elif args.h_type == 'fcc':
        args.nconcepts +=     int(not args.nobias)
        conceptizer  = image_fcc_conceptizer(11, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)
    else:
        raise ValueError('Unrecognized h_type')

    args.theta_arch = "dummy"
    model_path, log_path, results_path = generate_dir_names('cancer', args)


    parametrizer = dfc_parametrizer(input_dim, *layer_sizes, args.nconcepts, args.theta_dim)

    aggregator   = additive_scalar_aggregator(args.concept_dim,args.nclasses)

    model        = GSENN(conceptizer, parametrizer, aggregator)#, learn_h = args.train_h)

    if args.theta_reg_type == 'unreg':
        trainer = VanillaClassTrainer(model, args)
    elif args.theta_reg_type == 'grad1':
        trainer = GradPenaltyTrainer(model, args, typ = 1)
    elif args.theta_reg_type == 'grad2':
        trainer = GradPenaltyTrainer(model, args, typ = 2)
    elif args.theta_reg_type == 'grad3':
        trainer = GradPenaltyTrainer(model, args, typ = 3)
    elif args.theta_reg_type == 'crosslip':
        trainer = CLPenaltyTrainer(model, args)
    else:
        raise ValueError('Unrecoginzed theta_reg_type')


    trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)

    trainer.plot_losses(save_path=results_path)

    # Load Best One
    checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'))
    model = checkpoint['model']
    model        = GSENN(conceptizer, parametrizer, aggregator)#, learn_h = args.train_h)


    model = Wrapper(model)
    
    
    x = x_test[0, :]
    print("Verifying Explanation Setup:")
    print("Using model.predict() - ", model.predict(x))
    print("Using explanation - ",  pred_from_coefs(model.explain(x), x))
    
    results = {}

    test_acc = trainer.validate(test_loader, fold = 'test').data.numpy().reshape((1))[0]
    results['acc']  = np.float64(test_acc) / 100
    print('Test accuracy: {:8.2f}'.format(test_acc))
    
    standard, causal, stability = metrics_self(model, x_test)
        
    results['standard'] = np.float64(standard[0])
    results['causal'] = np.float64(causal[0,0])
    results['stability'] = np.float64(stability)
    
    print(results)
    with open("out.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()



