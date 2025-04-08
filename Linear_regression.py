import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns
import matplotlib.pyplot as plt

cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
print(cars.head())

sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)
#plt.show()

w = torch.rand(1, requires_grad=True, dtype=torch.float32)
b = torch.rand(1, requires_grad=True, dtype=torch.float32)

X_list = cars['wt'].to_list()
Y_list = cars['mpg'].to_list()

X_np = np.array(X_list, dtype= np.float32).reshape(-1,1)
Y_np = np.array(Y_list, dtype= np.float32).reshape(-1,1)

print(X_np.shape)
print(Y_np.shape)

X = torch.from_numpy(X_np)
Y = torch.from_numpy(Y_np)

print(X)
print(Y)


# X = torch.tensor(X_list)
# Y = torch.tensor(Y_list)

num_epochs = 100
learning_rate = 0.0001

for epoch in range(num_epochs):
    for i in range(len(X)):  # Stochastic Gradient Descent
        y_pred = X[i] * w + b
        loss_tensor = (y_pred - Y[i]) ** 2
        loss_tensor = loss_tensor.sum()  # Ensure it's scalar

        loss_tensor.backward()
        loss_value = loss_tensor.item()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # Always zero out gradients outside no_grad
        w.grad.zero_()
        b.grad.zero_()

    print(f"Epoch {epoch+1}, Loss: {loss_value}")

print(f"\nFinal Weight: {w.item()}, Bias: {b.item()}")

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_np, Y_list)
print(f"Slope: {reg.coef_}, Bias: {reg.intercept_}")

from torchviz import make_dot
make_dot(loss_tensor).render("test_graph", format="png")
