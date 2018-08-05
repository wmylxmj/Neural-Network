# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 13:30:51 2018

@author: wmy
"""

import numpy as np
import tensorflow as tf
import scipy.special
import matplotlib.pyplot as plt 

'''neural network class definition'''
class NeuralNetwork:
    
    '''initialise the neural network'''
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate): 
        
        #set number of nodes in each input, hidden, output layers
        self.input_nodes = inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes  
        
        #learning rate
        self.learning_rate = learningrate
        
        #weights
        self.W_input_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), 
                                               (self.hidden_nodes, self.input_nodes))
        self.W_hidden_output = np.random.normal(0.0, pow(self.output_nodes, -0.5), 
                                                (self.output_nodes, self.hidden_nodes))
        
        #active function sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)   
        
        pass
    
    '''train the neural network'''
    def train(self, inputslist, targetslist):
        
        #targets
        Targets = np.array(targetslist, ndmin=2).T
        #convert inputs list to 2d array
        Input_inputs = np.array(inputslist, ndmin=2).T
        #input layer's outputs
        Input_outputs = Input_inputs
                
        #X_hidden = W_input_hidden * Input
        Hidden_inputs = np.dot(self.W_input_hidden, Input_outputs)
        #active the signals
        Hidden_outputs = self.activation_function(Hidden_inputs)
        
        #output layer's input signal
        Output_inputs = np.dot(self.W_hidden_output, Hidden_outputs)
        #final outputs
        Output_outputs = self.activation_function(Output_inputs)
        
        #output layer errors
        Output_errors = Targets - Output_outputs
        #hidden layer errors
        Hidden_errors = np.dot(self.W_hidden_output.T, Output_errors)
        
        #update the weights for the links between the hidden and output layers
        self.W_hidden_output += self.learning_rate * np.dot((Output_errors * Output_outputs *
                                                             (1.0 - Output_outputs)), np.transpose(Hidden_outputs))
        #update the weights fot the links between the input and hidden layers
        self.W_input_hidden += self.learning_rate * np.dot((Hidden_errors * Hidden_outputs *
                                                            (1.0 - Hidden_outputs)), np.transpose(Input_outputs))
        pass
    
    '''query the neural network'''
    def query(self, inputslist):
        
        #convert inputs list to 2d array
        Input_inputs = np.array(inputslist, ndmin=2).T
        #input layer's outputs
        Input_outputs = Input_inputs
        
        #X_hidden = W_input_hidden * Input
        Hidden_inputs = np.dot(self.W_input_hidden, Input_outputs)
        #active the signals
        Hidden_outputs = self.activation_function(Hidden_inputs)
        
        #output layer's input signal
        Output_inputs = np.dot(self.W_hidden_output, Hidden_outputs)
        #final outputs
        Output_outputs = self.activation_function(Output_inputs)
        
        return Output_outputs
    
n = NeuralNetwork(3, 25, 3, 2.0)
for i in range(50):
    n.train([0.0, 0.0, 0.0],[0.9, 0.1, 0.5])
    print(n.query([0.0, 0.0, 0.0]))
    
