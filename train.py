import os
import sys
import json
import numpy as np
from importlib import import_module
from networks import NeuralNetwork
from json_tricks import load
name = sys.argv[1]
layers = import_module("saved."+name+".structure").layers
hyperparams = json.loads(open(os.getcwd()+"/saved/"+name+"/hyperparams.json").read())
training_data = load('saved/'+name+'/training_data.json')
testing_data = load('saved/'+name+'/testing_data.json')
nn = NeuralNetwork(name,layers,**hyperparams)
nn.load()
nn.train(training_data,testing_data)
