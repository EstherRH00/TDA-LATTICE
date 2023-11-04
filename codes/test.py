import torch

# Load the file
pt_file = torch.load("../data/sports/5-core/image_adj_10.pt")

# Print the head of the file
print(pt_file[:5])