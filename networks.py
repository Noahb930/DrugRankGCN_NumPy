import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
from random import shuffle

class NeuralNetwork:
    def __init__(self,name,layers,loss_func,lr,momentum,mini_batch_size):
        self.name = name
        self.layers = layers
        self.training_loss = np.array([])
        self.testing_loss = np.array([])
        self.training_accuracy = np.array([])
        self.testing_accuracy = np.array([])
        self.loss_func = loss_func
        self.lr = lr
        self.momentum = momentum
        self.mini_batch_size = mini_batch_size
    def feedforward(self,input):
        layer_inputs = input
        for layer_input in layer_inputs:
            layer_input["readout"] = []
        for layer in self.layers:
            layer_inputs = layer.feedforward(layer_inputs)
        return layer_inputs
    def calculate_loss(self,outputs,ranks):
        loss_array = []
        for output1, rank1 in zip(outputs,ranks):
            loss_array.append([])
            for output2, rank2 in zip(outputs, ranks):
                Phat = 1.0 if rank1>rank2 else 0.0 if rank2>rank1 else 0.5
                P  = np.clip(1/(1+np.exp(output2-output1)),0.000000001,0.99999999)
                loss_array[-1].append(-1*(Phat*np.log(P)+(1-Phat)*np.log(1-P)))
        return np.mean(loss_array)
    def calculate_loss_deriv(self,outputs, ranks):
        dLoss_array = []
        for output1, rank1 in zip(outputs,ranks):
            dLoss = np.array([[0.0]])
            for output2, rank2 in zip(outputs, ranks):
                Phat = 1.0 if rank1>rank2 else 0.0 if rank2>rank1 else 0.5
                P  = np.clip(1/(1+np.exp(output2-output1)),0.000000001,0.99999999)
                dLoss += (P-Phat)
            dLoss_array.append(dLoss/len(outputs))
        return dLoss_array
    def calculate_accuracy(self,outputs,ranks):
        accuracy_array = []
        for output1, rank1 in zip(outputs,ranks):
            correct = []
            for output2, rank2 in zip(outputs, ranks):
                Phat = 1.0 if rank1>rank2 else 0.0 if rank2>rank1 else 0.5
                P  = np.clip(1/(1+np.exp(output2-output1)),0.000000001,0.99999999)
                correct.append(1 if np.abs(P-Phat)<0.2 else 0)
            accuracy_array.append(sum(correct) / len(correct))
        return np.mean(accuracy_array)
    def backprop(self,outputs,ground_truths):
        layer_inputs = (self.calculate_loss_deriv(outputs,ground_truths),)
        for layer in reversed(self.layers):
            layer_inputs = layer.backprop(self.lr,self.momentum,*layer_inputs)
    def train(self,training_data,testing_data):
        num_mini_batches = int(len(training_data)/self.mini_batch_size)
        while True:
            try:
                shuffle(training_data)
                mini_batches = np.array_split(training_data,num_mini_batches)
                for mini_batch in mini_batches:
                    ranks = [compound["rank"] for compound in mini_batch]
                    outputs = self.feedforward(mini_batch)
                    self.backprop(outputs,ranks)
                    training_loss = self.calculate_loss(outputs,ranks)
                    training_accuracy = self.calculate_accuracy(outputs,ranks)
                    self.training_loss = np.append(self.training_loss,[training_loss])
                    self.training_accuracy = np.append(self.training_accuracy,[training_accuracy])
                    outputs = self.feedforward(testing_data)
                    ranks = [compound["rank"] for compound in testing_data]
                    testing_loss = self.calculate_loss(outputs,ranks)
                    testing_accuracy = self.calculate_accuracy(outputs,ranks)
                    self.testing_loss = np.append(self.testing_loss,[testing_loss])
                    self.testing_accuracy = np.append(self.testing_accuracy,[testing_accuracy])
                    print("Training Loss: "+str(training_loss)+" Training Accuracy: "+str(training_accuracy)+" Testing Loss: "+str(testing_loss)+" Testing Accuracy: "+str(testing_accuracy))
            except KeyboardInterrupt:
                self.save()
                print("Stopped training network "+self.name+" and saved current state")
                break
    def reset(self):
        for layer in self.layers:
            print(layer)
            layer.reset()
    def save(self):
        np.savez(os.getcwd()+"/saved/"+self.name+"/learning_curves.npz",training_loss=self.training_loss,testing_loss=self.testing_loss,training_accuracy=self.training_accuracy, testing_accuracy = self.testing_accuracy)
        for layer in self.layers:
            layer.save(self.name)
    def load(self):
        learning_curves = np.load(os.getcwd()+"/saved/"+self.name+"/learning_curves.npz")
        self.training_loss = learning_curves["training_loss"]
        self.testing_loss = learning_curves["testing_loss"]
        self.training_accuracy = learning_curves["training_accuracy"]
        self.testing_accuracy = learning_curves["testing_accuracy"]
        for layer in self.layers:
            layer.load(self.name)
    def graph(self,window):
        moving_training_loss_average = []
        moving_testing_loss_average = []
        moving_training_accuracy_average = []
        moving_testing_accuracy_average = []
        for i in range(self.training_loss.shape[0]-window):
            moving_training_loss_average.append(np.mean(self.training_loss[i:i+window]))
            moving_testing_loss_average.append(np.mean(self.testing_loss[i:i+window]))
            moving_training_accuracy_average.append(np.mean(self.training_accuracy[i:i+window]))
            moving_testing_accuracy_average.append(np.mean(self.testing_accuracy[i:i+window]))
        fig, (loss,accuracy) = plt.subplots(2, constrained_layout=True)
        fig.suptitle("Training of "+self.name)
        iterations = np.arange(self.training_loss.shape[0] - window)
        loss.plot(iterations,moving_training_loss_average,label="Training Data")
        loss.plot(iterations,moving_testing_loss_average,label="Testing Data")
        loss.set_xlabel('Iterations')
        loss.set_ylabel('Loss')
        loss.legend()
        accuracy.plot( iterations,moving_training_accuracy_average,label = "Training Data")
        accuracy.plot( iterations,moving_testing_accuracy_average,label = "Testing Data")
        accuracy.set_xlabel('Iterations')
        accuracy.set_ylabel('Accuracy')
        accuracy.legend()
        plt.show()
