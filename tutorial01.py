import torch
import numpy as np
'''
data = [[1,2],[3,4]]
t_data = torch.tensor(data)

tensor = torch.rand(3,4)

print(f"RandomTesor: {tensor}")
print(f"Device: {tensor.device}")

if torch.cuda.is_available():
    tensor2 = tensor.to('cuda')

shape = (2,3,)
rand_t = torch.rand(shape)
ones_t = torch.ones(shape)
zeros_t = torch.zeros(shape)

tensor2 = torch.ones(3,4)
print(f"Device: {tensor2.device}")
print('First R:', tensor2[0])
print('first C:', tensor2[:,0])
print('Last C:',tensor2[...,-1])

tensor2[:,1]=2
print(f"tensor*tensor2 \n {tensor*tensor2}\n")
'''
t = torch.ones(2,5)
print(f"t:{t}\n")

n = t.numpy()
t.add_(1)
print(f"n:{n}\n")

np.add(n, 2, out=n)
t2 = torch.from_numpy(n)
print(f"from numpy:{t2}\n")
print(f"from numpy:{t2.dtype}\n")