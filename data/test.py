import numpy as np

def compare_npy_files(file1, file2):
    # Load the arrays from the .npy files
    array1 = np.load(file1)
    array2 = np.load(file2)
    print("SHAPES:", array1.shape, array2.shape)

# Example usage:
file1_path = "./Musical_Instruments_2/image_feat.npy"
file2_path = "./Musical_Instruments_2/image_feat_TDA.npy"
compare_npy_files(file1_path, file2_path)

# Example usage:
file1_path = "./Musical_Instruments_2/text_feat.npy"
file2_path = "./Musical_Instruments_2/text_feat_TDA.npy"
compare_npy_files(file1_path, file2_path)