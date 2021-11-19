from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
import os
import random
import pubchempy as pcp
from json_tricks import dump
from PIL import Image

file_path = os.getcwd()+"/compounds.csv"
test_decimal = 0.3333333
num_permutations = 20

def generate_adjacency_matrix(mol):
    adjacency_matrix = np.zeros((mol.GetNumAtoms(),mol.GetNumAtoms()))
    for bond in mol.GetBonds():
        adjacency_matrix[bond.GetBeginAtomIdx()][bond.GetEndAtomIdx()] = 1
        adjacency_matrix[bond.GetEndAtomIdx()][bond.GetBeginAtomIdx()] = 1
    return adjacency_matrix

def generate_vocabulary(mols):
    arry = []
    for mol in mols:
        for atom in mol.GetAtoms():
            arry.append(atom.GetSymbol())
    unique_arry = list(set(arry))
    return unique_arry

def generate_feature_matrix(mol,vocab):
    output = np.zeros((mol.GetNumAtoms(),len(vocab)))
    for i, atom in enumerate(mol.GetAtoms()):
        output[i][vocab.index(atom.GetSymbol())]=1
    return output

def shuffle_adjacency_matrix(adjacency_matrix,order):
    shuffled_matrix = np.zeros(adjacency_matrix.shape)
    for bond in np.transpose(np.nonzero(adjacency_matrix)):
        shuffled_matrix[order[bond[0]]][order[bond[1]]] = 1
        shuffled_matrix[order[bond[1]]][order[bond[0]]] = 1
    return shuffled_matrix

def shuffle_feature_matrix(feature_matrix,order):
    shuffled_matrix = np.zeros(feature_matrix.shape)
    for i,row in enumerate(feature_matrix):
        shuffled_matrix[i] = feature_matrix[order[i]]
    return shuffled_matrix

def encode(data,vocab,size,num_permutations):
    encoded_data = []
    for name, mol, label, rank in data:
        base_adjacency_matrix = generate_adjacency_matrix(mol)
        base_feature_matrix = generate_feature_matrix(mol,vocab)
        permutations = [np.random.permutation(mol.GetNumAtoms()) for i in range(num_permutations)]
        padding = 0
        #padding = size - base_adjacency_matrix.shape[0]
        for permutation in permutations:
            shuffled_feature_matrix = shuffle_feature_matrix(base_feature_matrix,permutation)
            shuffled_adjacency_matrix = shuffle_adjacency_matrix(base_adjacency_matrix,permutation)
            padded_feature_matrix = np.pad(shuffled_feature_matrix,((0,padding),(0,0)),"constant")
            padded_adjacency_matrix = np.pad(shuffled_adjacency_matrix,((0,padding),(0,padding)),"constant")
            encoded_data.append({"name":name,"adjacency_matrix":padded_adjacency_matrix,"feature_matrix":padded_feature_matrix,"label":label,"rank":rank})
    return encoded_data

file = pd.read_csv(file_path)
mols = [Chem.MolFromSmiles(row["SMILES"])for i, row in file.iterrows()]
names = [row["Name"] for i, row in file.iterrows()]
labels = [np.array([[row["TNF-alpha"]],[row["MTT"]]]) for i, row in file.iterrows()]
ranks = [i for i, row in file.iterrows()]

vocabulary = generate_vocabulary(mols)
data = list(zip(names,mols,labels,ranks))
random.shuffle(data)
split = int(test_decimal*len(data))
testing_data = data[:split]
training_data = data[split:]
encoded_training_data = encode(training_data,vocabulary,50,num_permutations)
encoded_testing_data = encode(testing_data,vocabulary,50,num_permutations)

with open('training_data.json','w') as write_file:
    dump(encoded_training_data, write_file)
with open('testing_data.json','w') as write_file:
    dump(encoded_testing_data, write_file)

