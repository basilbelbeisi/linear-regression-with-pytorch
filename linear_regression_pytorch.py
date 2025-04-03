# Linear Regression with PyTorch - Full Code Example
# From the article: https://dataskillblog.com/linear-regression-with-pytorch
# Author: Basil Belbeisi
# This code demonstrates how gradient descent updates slope (w) and bias (b)
# to minimize the prediction error for a simple linear regression task

import torch
import matplotlib.pyplot as plt

# Step 1: Define the actual value we want to predict (target output)
# Let's say the true electricity bill for usage x = 3.0 is y = 6.0
x_input = torch.tensor([[3.0]])
y_actual = torch.tensor([[6.0]])

# Step 2: Initialize model parameters w (slope) and b (bias)
# These are our "guesses" that the model will improve
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)

# Step 3: Define the forward pass (the prediction function)
def forward(x):
    return w * x + b

# Step 4: Set the learning rate
# This controls how big each update step is during training
learning_rate = 0.01

# Step 5: Track values for plotting
w_values = []       # Store w after each epoch
b_values = []       # Store b after each epoch
loss_values = []    # Store loss after each epoch

# Step 6: Train for 20 epochs using gradient descent
for epoch in range(20):
    # Forward pass: predict the output using current w and b
    y_pred = forward(x_input)

    # Compute loss using Mean Squared Error: (y_actual - y_predicted)^2
    loss = (y_actual - y_pred).pow(2).mean()

    # Backward pass: compute gradients (∂loss/∂w and ∂loss/∂b)
    loss.backward()

    # Gradient descent update: adjust w and b in opposite direction of gradients
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Store values to track learning progress
    w_values.append(w.item())
    b_values.append(b.item())
    loss_values.append(loss.item())

    # Reset gradients for the next iteration
    w.grad.zero_()
    b.grad.zero_()

# Step 7: Print final values
print("Final w:", w.item())
print("Final b:", b.item())
print("Final prediction for x = 3.0:", forward(x_input).item())

# Step 8: Plot the learning progress of w, b, and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(range(1, 21), w_values)
plt.title("w over Epochs")
plt.xlabel("Epoch")
plt.ylabel("w")

plt.subplot(1, 3, 2)
plt.plot(range(1, 21), b_values, color="orange")
plt.title("b over Epochs")
plt.xlabel("Epoch")
plt.ylabel("b")

plt.subplot(1, 3, 3)
plt.plot(range(1, 21), loss_values, color="red")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()
