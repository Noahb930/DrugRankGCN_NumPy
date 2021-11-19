from networks import NeuralNetwork
from layers import ReadoutLayer,SortPoolingLayer, FullyConnectedLayer, GraphConvolutionLayer, ConcatenationLayer, AveragePoolingLayer,  BatchNormalizationLayer
import numpy as np

layers = []

layers.append(GraphConvolutionLayer(6,32,"leaky_relu",False))
layers.append(GraphConvolutionLayer(32,32,"leaky_relu",False))
layers.append(GraphConvolutionLayer(32,32,"leaky_relu",False))
layers.append(GraphConvolutionLayer(32,32,"leaky_relu",False))
layers.append(GraphConvolutionLayer(32,32,"leaky_relu",False))
layers.append(GraphConvolutionLayer(32,32,"leaky_relu",False))
layers.append(ReadoutLayer())
layers.append(SortPoolingLayer(12))
layers.append(FullyConnectedLayer(2304,4096,"leaky_relu",False))
layers.append(FullyConnectedLayer(4096,2048,"leaky_relu",False))
layers.append(FullyConnectedLayer(2048,256,"leaky_relu",False))
layers.append(FullyConnectedLayer(256,1,"identity",False))


