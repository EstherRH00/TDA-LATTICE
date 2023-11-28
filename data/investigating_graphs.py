import torch
import numpy as np

folder = './Musical_Instruments'

images = np.load(folder + '/image_feat.npy')
print("Images, inicialment", images.shape)
print(images[:5])


image_1 = torch.load(folder + "/5-core/image_1.pt")
image_2 = torch.load(folder + "/5-core/image_2.pt")
image_3 = torch.load(folder + "/5-core/image_3.pt")

print("Images, matriu de similitud", image_1.shape)
print(image_1[:5])
print("Images, knn de la matriu de similitud", image_2.shape)
print(image_2[:5])

id = torch.eye(image_2.shape[0])
is_identity = torch.equal(image_2, id)
print('Es la identitat? ', is_identity)
if not is_identity:
    # Find a row that is different
    different_row_idx = torch.unique(torch.nonzero(image_2 - id)[:, 0])

    new_mat = image_2[different_row_idx]
    np.savetxt('./diff_rows.txt', new_mat.numpy())

print("Images, laplaciana normalitzada - adjacencia final", image_3.shape)
print(image_3[:5])

print('Es diagonal?', torch.all(image_3 == torch.diag(torch.diagonal(image_3))))

text_1 = torch.load(folder + "/5-core/text_1.pt")
text_2 = torch.load(folder + "/5-core/text_2.pt")
text_3 = torch.load(folder + "/5-core/text_3.pt")

print("texts, matriu de similitud", text_1.shape)
print(text_1[:5])
print("texts, knn de la matriu de similitud", text_2.shape)
print(text_2[:5])
print("texts, laplaciana normalitzada - adjacencia final", text_3.shape)
print(text_3[:5])


original = torch.load(folder + "/5-core/original_adj.pt")
print(original.shape)
print(original[:5])

graph_matrix = original.detach().numpy()
print('now is numpy', graph_matrix.shape)
print(graph_matrix[:5])
print(graph_matrix[-5:])

# Compute out-degrees of each node
out_degrees = np.sum(graph_matrix != 0, axis=1)

# Find the node with the highest and lowest out-degrees
most_connected_node = np.argmax(out_degrees)
least_connected_node = np.argmin(out_degrees)

# Compute the degree range
degree_range = np.max(out_degrees) - np.min(out_degrees)

# Find the nodes closest and farthest away from a given node (e.g., node 0)
source_node = 0
closest_nodes = np.argsort(graph_matrix[source_node])
farthest_nodes = np.argsort(graph_matrix[source_node])[::-1]

# Compute the node weight range
node_weights = np.diagonal(graph_matrix)
weight_range = np.max(node_weights) - np.min(node_weights)

# Exclude the diagonal for finding edge values
edges = graph_matrix[~np.eye(graph_matrix.shape[0], dtype=bool)]

# Find the maximum and minimum edge values
max_edge_value = np.max(edges)
min_edge_value = np.min(edges)

# Print the results
print(f"The most connected node is node {most_connected_node} with out-degree {np.max(out_degrees)}.")
print(f"The least connected node is node {least_connected_node} with out-degree {np.min(out_degrees)}.")
print(f"Degree range: {degree_range}.")
print(f"Node weight range: {weight_range}: [{np.min(node_weights)}, {np.max(node_weights)}]")
print(f"Edge value range: {max_edge_value - min_edge_value}: [{min_edge_value}, {max_edge_value}].")