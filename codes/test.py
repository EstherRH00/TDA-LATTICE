import torch
import numpy as np
your_matrix = torch.tensor([[1, 4, 0],
                           [5, 1, 3],
                           [7, 0, 1]])

identity_matrix = torch.eye(your_matrix.shape[0])

# Check if the matrix is the identity matrix
is_identity = torch.equal(your_matrix, identity_matrix)

if not is_identity:
    # Find a row that is different
    different_row_idx = torch.unique(torch.nonzero(your_matrix - identity_matrix)[:, 0])

    new_mat = your_matrix[different_row_idx]
    np.savetxt('./test.txt', new_mat.numpy())