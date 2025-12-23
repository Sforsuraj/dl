import numpy as np 
 
# Step 1: Define activation function 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
def sigmoid_derivative(x): 
    return x * (1 - x) 
 
# Step 2: Input dataset 
inputs = np.array([[0, 0, 1], 
                   [1, 1, 1], 
                   [1, 0, 1], 
                   [0, 1, 1]]) 
 
outputs = np.array([[0], [1], [1], [0]]) 
 
# Step 3: Initialize weights randomly 
np.random.seed(1) 
weights = 2 * np.random.random((3, 1)) - 1 
print("Beginning Randomly Generated Weights:\n", weights) 
 
# Step 4: Train the model 
for iteration in range(20000): 
    input_layer = inputs 
    outputs_pred = sigmoid(np.dot(input_layer, weights)) 

 
    error = outputs - outputs_pred 
    adjustments = error * sigmoid_derivative(outputs_pred) 
    weights += np.dot(input_layer.T, adjustments) 
 
print("\nEnding Weights After Training:\n", weights) 
 
# Step 5: Test model 
new_input = np.array([1, 0, 0]) 
output = sigmoid(np.dot(new_input, weights)) 
print("\nConsidering New Situation:", new_input) 
print("New Output Data:\n", output) 