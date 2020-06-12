import numpy as np
import torch
#temp,rainfall,humidity
inputs = np.array([[73,67,43],
                   [91,88,64],
                   [87,134,68],
                   [102,43,37],
                   [69,96,70]],dtype = 'float32')


targets = np.array([[56,70],
                    [81,101],
                    [119,133],
                    [22,37],
                    [103,119]],dtype = 'float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

#
w = torch.randn(2,3,requires_grad = True)
bias = torch.randn(2,requires_grad = True)

def model(x):
    return x@w.t() + bias

def mse(t1,t2):
    diff = t1-t2
    return torch.sum(diff*diff)/diff.numel()
for i in range(10000):
    preds = model(inputs)
    loss = mse(preds,targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad* 0.00001
        bias-= bias.grad* 0.00001
        w.grad.zero_()
        bias.grad.zero_()
        
preds = model(inputs)
loss = mse(preds,targets)
print(loss)
print(preds)
print(targets)

        