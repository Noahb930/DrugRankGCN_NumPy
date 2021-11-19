import os
import sys
import json
import numpy as np
from networks import NeuralNetwork
from structure import layers
from json_tricks import dump, load

name = sys.argv[1]

try:
    os.mkdir(os.getcwd()+"/saved/"+name)
    os.mkdir(os.getcwd()+"/saved/"+name+"/trainable_params")
except:
    print("Network "+name+" already exists")
    sys.exit()

hyperparams_input = open(os.getcwd()+"/hyperparams.json").read()
hyperparams_json = json.loads(hyperparams_input)

nn = NeuralNetwork(name,layers,**hyperparams_json)
nn.save()

structure_input = open(os.getcwd()+"/structure.py").read()
hyperparams_output = open(os.getcwd()+"/saved/"+name+"/hyperparams.json","w")
structure_output = open(os.getcwd()+"/saved/"+name+"/structure.py","w")
hyperparams_output.write(hyperparams_input)
structure_output.write(structure_input)

open(os.getcwd()+'/saved/'+name+'/training_data.json','w').write(open(os.getcwd()+'/training_data.json').read())
open(os.getcwd()+'/saved/'+name+'/testing_data.json','w').write(open(os.getcwd()+'/testing_data.json').read())



