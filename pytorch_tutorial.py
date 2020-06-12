from __future__ import print_function
import torch

#Print a uninitialized matrix
x = torch.empty(5, 3)
#print(x)
#print a random matrix

x = torch.rand(5,3)
#print(x)

#print a matrix of zeros

x = torch.zeros(5, 3, dtype=torch.long)
#print(x)

x = torch.tensor([5.5, 3])
#print(x)


x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
#print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
#print(x)  

#print(x.size())

#Operations
y = torch.randn(5,3)
#print(x+y)

#print(torch.add(x, y))  #2nd method

result = torch.empty(5, 3) 
torch.add(x, y, out=result)  #storing out in result
#print(result)

y.add_(x)
#print(y)

# y = print(y[:, 1])
# print(y)
#Reshaping
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
# print("x = ",x)
# print("y = ",y)
# print("z = ",z)
# print(x.size(), y.size(), z.size())
#If you have one number you may use item

x = torch.randn(1)
# print(x)
# print(x.item())

#Torch to numpy
# a = torch.ones(5)
# print(a)

# b = a.numpy()
# print(b)

# a.add_(1)
# print(a)
# print(b)

#Converting numpy to torch tensor
# import numpy as np
# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)    
    print(z.to("cpu", torch.double)) 




