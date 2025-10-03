from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#import the basic modules and the dataset.
data = fetch_california_housing(as_frame=True)
X = data.data.to_numpy()
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = X.T   
Y = data.target.to_numpy()   
Y_mean = Y.mean()
Y_std = Y.std()
Y = (Y - Y_mean) / Y_std 
Y = Y.reshape(1 , -1)
# the above block of code loads the dataset and does the standardization 
n1 = 30
n2 = 20
n3 = 1

def tanh(z):
    return np.tanh(z)

def sigmoid(z):
    z = np.clip(z , -500 , 500)
    return 1 / (1 + np.exp(-z))

def deriv_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh_deriv(z):
    return 1 - np.tanh(z)**2


def params_init(n1 , n2 , n3):
    parameters = {
        "W1":np.random.randn(n1 , X.shape[0]),
        "b1":np.zeros((n1 , 1)),
        "W2":np.random.randn(n2 , n1),
        "b2":np.zeros((n2 ,1)),
        "W3":np.random.randn(n3 , n2),
        "b3":np.zeros((n3 , 1))
    }
    return parameters

def forward_prop(X,parameters):
    Z1 = np.dot(parameters["W1"] , X) + parameters["b1"]
    A1 = sigmoid(Z1)
    Z2 = np.dot(parameters["W2"] , A1) + parameters["b2"]
    A2 = sigmoid(Z2)
    Z3 = np.dot(parameters["W3"] , A2) + parameters["b3"]
    output = tanh(Z3)
    cache = (Z1 , A1 , Z2 , A2 , Z3)    
    return output , cache

def back_prop(X , Y , parameters , cache , output):
    m = X.shape[1]
    Z1 , A1 , Z2 , A2 , Z3 = cache

    dz3 = (output - Y) * tanh_deriv(Z3)
    dw3 = (1 / m) * np.dot(dz3 , A2.T)
    db3 = (1 / m) * np.sum(dz3)
    dA2 = np.dot(parameters["W3"].T,dz3)
    #grads from the 3rd layer 

    dz2 = deriv_sigmoid(Z2) * dA2
    dw2 = (1/m) * np.dot(dz2 , A1.T)
    db2 = (1/m) * np.sum(dz2)
    dA1 = np.dot(parameters["W2"].T,dz2)

    #grads from 2nd layer

    dz1 = deriv_sigmoid(Z1) * dA1
    dw1 = (1/m) * np.dot(dz1 , X.T)
    db1 = (1/m) * np.sum(dz1)

    grads = {
        "dW1": dw1, "db1": db1,
        "dW2": dw2, "db2": db2,
        "dW3": dw3, "db3": db3
    }

    return grads
def update_parameters(grads , parameters):
    parameters["W1"] -= 0.01 * grads["dW1"]
    parameters["b1"] -= 0.01 * grads["db1"]
    parameters["W2"] -= 0.01 * grads["dW2"]
    parameters["b2"] -= 0.01 * grads["db2"]
    parameters["W3"] -= 0.01 * grads["dW3"]
    parameters["b3"] -= 0.01 * grads["db3"]
    return parameters

parameters = params_init(n1 , n2 , n3)

epochs = 10
mse_list = []
y_true = Y.flatten()



for epoch in range(epochs):
    indices = np.random.permutation(X.shape[1])# a random shuffle of training examples
    for i in indices:
        x_i = X[:, i].reshape(-1, 1)
        y_i = Y[:, i].reshape(1, 1)
        output, cache = forward_prop(x_i, parameters)
        grads = back_prop(x_i, y_i, parameters, cache, output)
        loss = np.mean((output - y_i) ** 2)
        mse_list.append(loss)  
        parameters = update_parameters(grads, parameters)

    if (epoch + 1) % 10 == 0:    
        output_full, _ = forward_prop(X, parameters)
        y_pred_full = output_full.flatten()
        mse = mean_squared_error(y_true, y_pred_full)
        mse_list.append(mse)


epoch_steps = [i for i in range(1 , epochs + 1) if i % 10 == 0]  

# cost vs epoch plot
step = 1000
downsampled = mse_list[::step]
plt.figure(figsize=(12, 4))
plt.plot(downsampled)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (Loss)")
plt.title("Training Loss vs Epochs")
plt.grid(True)
plt.show()

output_full, _ = forward_prop(X, parameters)
y_pred = output_full.flatten()

# the regression plot

plt.figure(figsize=(8, 5))
plt.scatter(y_true, y_pred, alpha=0.5, color='teal')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Diagonal line
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)

plt.show()

