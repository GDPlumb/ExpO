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

sys.path.insert(0, "../../Code/")
from ExplanationMetrics import metrics_maple, metrics_lime


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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)
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
        self.index = 0
        
    def predict(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        x_torch = Variable(torch.from_numpy(x)).float()
        pred = self.model(x_torch)
        pred_ran = pred.cpu()
        result = pred_ran.data.numpy()
        return result
        
    def set_index(self, i):
        self.index = i

    def predict_index(self, x):
        return np.squeeze(self.predict(x)[:, self.index])
            
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


    results = {}


    test_acc = trainer.validate(test_loader, fold = 'test').data.numpy().reshape((1))[0]
    results['acc']  = np.float64(test_acc)
    print('Test accuracy: {:8.2f}'.format(test_acc))
    
    wrapper = Wrapper(model)
    
    r_maple = np.float64(metrics_maple(wrapper, x_train, x_valid, x_test))
    print('MAPLE', r_maple)
    results['maple_pf'] = r_maple[0][0]
    results['maple_nf'] = r_maple[1][0]
    results['maple_s'] = r_maple[2][0]
    
    r_lime = np.float64(metrics_lime(wrapper, x_train, x_test))
    print('LIME', r_lime)
    results['lime_pf'] = r_lime[0][0]
    results['lime_nf'] = r_lime[1][0]
    results['lime_s'] = r_lime[2][0]
    
    
    print(results)
    with open("out.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()


