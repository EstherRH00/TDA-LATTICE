import torch
import numpy as np

# Example weighted adjacency matrix
graph_matrix = np.array([[2, 0, 3, 0],
                         [0, 0, 0, 4],
                         [0, 0, 0, 0],
                         [0, 5, 0, 0]])

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

# Print the results
print(f"The most connected node is node {most_connected_node} with out-degree {np.max(out_degrees)}.")
print(f"The least connected node is node {least_connected_node} with out-degree {np.min(out_degrees)}.")
print(f"Degree range: {degree_range}.")
print(f"Closest nodes to node {source_node}: {closest_nodes}.")
print(f"Farthest nodes from node {source_node}: {farthest_nodes}.")
print(f"Node weight range: {weight_range}.")