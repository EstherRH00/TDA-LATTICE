import torch
import numpy as np

# Create a torch matrix with 3 rows and 5 columns
torch_matrix = torch.rand((3, 5))

# Create a numpy vector with 4 values
v1 = np.array([1, 2, 3, 4])
v2 = np.array([1, 2])

print(v1.shape)
print(v2.shape)

print(np.concatenate((v1, v2), axis=0))

python main.py --dataset Musical_Instruments --model lattice
python main.py --dataset Musical_Instruments --model lattice
python main.py --dataset Musical_Instruments --model lattice
python main.py --dataset Musical_Instruments --model lattice
python main.py --dataset Musical_Instruments --model lattice
python main.py --dataset Digital_Music --model lattice
python main.py --dataset Digital_Music --model lattice
python main.py --dataset Digital_Music --model lattice
python main.py --dataset Digital_Music --model lattice
python main.py --dataset Digital_Music --model lattice
