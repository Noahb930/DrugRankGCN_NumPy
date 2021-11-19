import os
import sys
import json
import numpy as np
from importlib import import_module
from networks import NeuralNetwork
from json_tricks import load
name = sys.argv[1]
window = int(sys.argv[2])
layers = import_module("saved."+name+".structure").layers
hyperparams = json.loads(open(os.getcwd()+"/saved/"+name+"/hyperparams.json").read())
data = load('saved/'+name+'/training_data.json')
nn = NeuralNetwork(name,layers,**hyperparams)
nn.load()
nn.graph(window)
