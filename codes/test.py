import torch
import numpy as np

# Create a torch matrix with 3 rows and 5 columns
torch_matrix = torch.rand((3, 5))

# Create a numpy vector with 4 values
np_vector = np.array([1, 2, 3, 4])

# Convert the numpy vector to a torch tensor
torch_vector = torch.tensor(np_vector)

print(len(torch_vector))

# Repeat the torch vector to match the number of columns (5) in the torch matrix
torch_vector_repeated = torch_vector.repeat(torch_matrix.size(0), 1)

# Concatenate the torch matrix and the repeated torch vector along the second axis (columns)
result_matrix = torch.cat((torch_matrix, torch_vector_repeated), dim=1)

print(result_matrix)

print(torch.cat((torch_vector, torch_vector)))
