import numpy as np
from scipy import linalg as spl
import os
import itertools

np.seterr(all='ignore')

class Layer:
    counter = itertools.count()
    def activate(self,input,type):
        if type == "relu":
            return np.where(input>0,input,0)    
        elif type == "leaky_relu":
            return np.where(input>0,input,input)    
        elif type == "sigmoid":
            output = 1/(1 + np.exp(input))
            output = np.clip(output,1e-12,0.99999999)
            return output
        elif type == "tanh":
            output = np.tanh(input)
            output = np.clip(output,-0.999999,0.999999)
            return output
        elif type == "identity":
            return input
    def activate_deriv(self,input,type):
        if type == "relu":
            return np.where(input>=0,1,0)
        if type == "leaky_relu":
            return np.where(input>=0,1,0.1)
        elif type == "sigmoid":
            return input * (1.0 - input)
        elif type == "tanh":
            output =  1.0 - np.square(np.tanh(input))
            return output
        elif type == "identity":
            return 1
    def save(self,dir):
        np.savez(os.getcwd()+"/saved/"+dir+"/layer"+str(self.id)+".npz",W=self.W,dW=self.dW,B=self.B,dB=self.dB)
    def load(self,dir):
        data = np.load(os.getcwd()+"/saved/"+dir+"/layer"+str(self.id)+".npz")
        self.W = data["W"]
        self.dW = data["dW"]
        self.B = data["B"]
        self.dB = data["dB"]
 
class FullyConnectedLayer(Layer):
    def __init__(self,input_dim,output_dim,activation_func,is_residual):
        self.id = next(self.counter)
        self.activation_func = activation_func
        self.W = np.random.randn(output_dim,input_dim)*0.3
        self.B = np.ones((output_dim,1))
        self.dW = np.zeros((output_dim,input_dim))
        self.dB = np.zeros((output_dim,1))
        self.is_residual = is_residual
    def feedforward(self,input):
        self.Xs = input
        self.Zs = [np.dot(self.W,X) + self.B + (X if self.is_residual else 0) for X in self.Xs]
        self.As = [self.activate(Z,self.activation_func) for Z in self.Zs]
        self.outputs = self.As
        return self.outputs
    def backprop(self,lr,momentum,dAs):
        self.dZs = [dA * self.activate_deriv(Z,self.activation_func) for dA, Z in zip(dAs,self.Zs)]
        self.dXs =np.clip([np.dot((self.W.T + 1 if self.is_residual else self.W.T),dZ) for dZ in self.dZs],-100,100)
        self.dW = momentum * self.dW + np.clip(np.mean([np.dot(dZ,X.T) for dZ, X in zip(self.dZs, self.Xs)],axis=0),-100,100)
        self.dB = momentum*self.dB + np.clip(np.mean([np.sum(dZ, axis=1,keepdims=True) for dZ in self.dZs],axis=0),-100,100)
        #print(self.dB)
        self.W -= lr * self.dW
        self.B -= lr * self.dB
        return (self.dXs,)
class GraphConvolutionLayer(Layer):
    def __init__(self,input_dim,output_dim,activation_func,is_residual):
        self.id = next(self.counter)
        self.activation_func = activation_func
        self.W = np.random.randn(input_dim,output_dim)*0.3
        self.B = np.ones((1,output_dim))
        self.dW = np.zeros((input_dim,output_dim))
        self.dB = np.zeros((1,output_dim))
        self.is_residual = is_residual
    def feedforward(self,input):
        self.As = [entry["adjacency_matrix"] for entry in input]
        self.A_hats = [A + np.eye(A.shape[0]) for A in self.As]
        self.D_inv_sqrts = [spl.sqrtm(np.linalg.pinv(np.diagflat(np.sum(A_hat, axis=0)))) for A_hat in self.A_hats]
        self.A_norms = [D_inv_sqrt.dot(A_hat).dot(D_inv_sqrt) for A_hat, D_inv_sqrt in zip(self.A_hats, self.D_inv_sqrts)]
        self.Fs = [entry["feature_matrix"] for entry in input]
        self.Zs = [A_norm.dot(F).dot(self.W)  + (F if self.is_residual else 0) for A_norm, F in zip(self.A_norms, self.Fs)]
        self.Hs = [self.activate(Z,self.activation_func) for Z in self.Zs]
        self.Rs = [entry["readout"] for entry in input]
        for R, H in zip(self.Rs, self.Hs):
            R.append(H)
        self.outputs = [{"adjacency_matrix":A,"feature_matrix":H,"readout":R} for A, H,R in zip(self.As,self.Hs,self.Rs)]
        return self.outputs
    def backprop(self,lr,momentum,dHs,dRs):
        self.dHs = [(dH + dR[-1])/2.0 for dH, dR in zip(dHs,dRs)]
        for dR in dRs:
            dR.pop()
        self.dZs = [np.clip(dH * self.activate_deriv(Z,self.activation_func),-100,100) for dH, Z in zip(self.dHs, self.Zs)]
        self.dFs = [np.clip(A_norm.T.dot(dZ).dot(self.W.T + 1 if self.is_residual else self.W.T),-100,100) for A_norm, dZ in zip(self.A_norms, self.dZs)]
        self.dW = momentum*self.dW + np.clip(np.mean([A_norm.dot(F).T.dot(dZ) for A_norm, F, dZ in zip(self.A_norms, self.Fs, self.dZs)], axis=0),-100,100)
        
#        self.dB = np.mean([np.sum(dZ, axis=0,keepdims=True) for dZ in self.dZs],axis=0)
        self.W -= lr * self.dW
 #       self.B -= lr * self.dB
        return (self.dFs,dRs)
class SortPoolingLayer(Layer):
    def __init__(self,output_dim):
        self.id = next(self.counter)
        self.output_dim=output_dim
    def feedforward(self,input):
        self.Fs = input
        self.orders = [np.argsort(-1 * F.T[0])[:self.output_dim] for F in self.Fs]
        self.Zs = [F[order] for F, order in zip(self.Fs,self.orders)]
        self.Hs = [Z.reshape(Z.shape[0]*Z.shape[1],1) for Z in self.Zs]
        
        self.outputs = self.Hs
        return self.outputs
    def backprop(self,lr,momentum,dHs):
        self.dZs = [dH.reshape(Z.shape[0],Z.shape[1]) for dH, Z in zip(dHs, self.Zs)]
        self.dFs = [np.zeros((F.shape[0],F.shape[1])) for F in self.Fs]
        self.indexes = [[(order.tolist().index(row) if row in order else None)for row in range(F.shape[0])] for order,F in zip(self.orders,self.Fs)]
        self.dFs = [np.vstack([(dZ[row] if row != None else np.zeros((1,dZ.shape[1]))) for row in index]) for dZ,index in zip(self.dZs,self.indexes)]
        return (self.dFs,)
    def load(self,dir):
        return
    def save(self,dir):
        return
class ConcatenationLayer(Layer):
    def __init__(self,input_dims):
        self.id = next(self.counter)
        self.input_dims = input_dims;
    def feedforward(self,input):
        self.Fs = [entry["feature_matrix"] for entry in input]
        self.Hs = [F.reshape(self.input_dims[0]*self.input_dims[1],1) for F in self.Fs]
        self.outputs = self.Hs
        return self.outputs
    def backprop(self,lr,momentum,dHs):
        self.dFs = [dH.reshape(self.input_dims) for dH in dHs]
        return self.dFs
    def load(self,dir):
        return
    def save(self,dir):
        return
class ReadoutLayer(Layer):
    def __init__(self):
        self.id = next(self.counter)
    def feedforward(self,input):
        self.Rs = [entry["readout"] for entry in input]
        self.Hs = [np.hstack(R) for R in self.Rs]
        return self.Hs
    def backprop(self,lr,momentum,dHs):
        self.dRs = []
        for R, dH in zip(self.Rs,dHs):
            dR = []
            split_idx = 0
            for F in R:
                dR.append(dH[:,split_idx:split_idx+F.shape[1]])
                split_idx += F.shape[1]
            self.dRs.append(dR)
        self.dFs = [dR[-1] for dR in self.dRs]
        return (self.dFs,self.dRs)
    def load(self,dir):
        return
    def save(self,dir):
        return

class AveragePoolingLayer(Layer):
    def __init__(self):
        self.id = next(self.counter)
    def feedforward(self,input):
        self.Fs = [entry["feature_matrix"] for entry in input]
        self.dims = [entry["feature_matrix"].shape[0] for entry in input]
        self.Hs = [np.mean(F,axis=0).reshape(F.shape[1],1) for F in self.Fs]
        self.outputs = self.Hs
        return self.outputs
    def backprop(self,lr,momentum,dHs):
        self.dFs = [np.tile(dH.T,(dim,1)) for dH, dim in zip(dHs, self.dims)]
        return self.dFs
    def load(self,dir):
        return
    def save(self,dir):
        return


class BatchNormalizationLayer(Layer):
    def __init__(self,input_shape,activation_func):
        self.id = next(self.counter)
        self.activation_func = activation_func
        self.beta = np.zeros(input_shape)
        self.gamma = np.ones(input_shape)
        self.dcdg = np.zeros(input_shape)
        self.dcdb = np.zeros(input_shape)
    def feedforward(self,input):
        self.x_s = np.array(input.Fs)
        self.mean = np.mean(self.x_s,axis = 0)
        self.pinv_var = 1/(np.mean((self.x_s - self.mean) ** 2, axis = 0)+ 1e-8)
        self.x_hats = (self.x_s - self.mean) * self.pinv_var
        self.z_s = [self.gamma*x_hat + self.beta for x_hat in self.x_hats]
        self.outputs = [self.activate(z,self.activation_func) for z in self.z_s]
        return self.outputs
    def backprop(self,lr,momentum,dcdo_s):
        self.dcdx_hats = [dcdo * self.gamma for dcdo in dcdo_s]
        self.dcdo_s = [((1. / len(self.dcdx_hats)) * self.pinv_var * (len(self.dcdx_hats)*dcdx_hat - np.sum(dcdx_hat, axis=0) - x_hat*np.sum(dcdx_hat*x_hat, axis=0)))/len(dcdo_s) for dcdx_hat, x_hat in zip(self.dcdx_hats, self.x_hats)]
        self.dcdg = np.sum(np.stack([dcdo*x_hat for dcdo, x_hat in zip(dcdo_s,self.x_hats)]), axis = 0)
        self.dcdb = np.sum(np.array(dcdo_s), axis=0)
        self.gamma-= lr * self.dcdg
        self.beta-= lr * self.dcdb
        return self.dcdo_s
    def save(self,dir):
        np.savez(os.getcwd()+"/saved/"+dir+"/layer"+str(self.id)+".npz",beta=self.beta,gamma=self.gamma,dcdg=self.dcdg,dcdb=self.dcdb)
    def load(self,dir):
        data = np.load(os.getcwd()+"/saved/"+dir+"/layer"+str(self.id)+".npz")
        self.beta = data["beta"]
        self.gamma = data["gamma"]
        self.dcdg = data["dcdg"]
        self.dcdb = data["dcdb"]
