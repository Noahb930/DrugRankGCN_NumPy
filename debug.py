import os
import sys
import json
import numpy as np
from importlib import import_module
from networks import NeuralNetwork

name = sys.argv[1]

hyperparams = json.loads(open(os.getcwd()+"/saved/"+name+"/hyperparams.json").read())
adjacency_matrixes = np.load("training_adjacency_matrixes.npz",allow_pickle=True)["arr_0"]
feature_matrixes = np.load("training_feature_matrixes.npz",allow_pickle=True)["arr_0"]
inputs =[{"feature_matrix":feature_matrix, "adjacency_matrix":adjacency_matrix} for adjacency_matrix, feature_matrix in zip(adjacency_matrixes, feature_matrixes)]
ground_truths = np.load("training_ground_truths.npz",allow_pickle=True)["arr_0"]

layers = import_module("saved."+name+".structure").layers
nn = NeuralNetwork(name,layers,hyperparams["loss_func"],hyperparams["lr"],hyperparams["momentum"],hyperparams["mini_batch_size"])
nn.load()

nn.debug(inputs,ground_truths)
