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

print('Es diagonal?', torch.all(image_3 == torch.diag(torch.diagonal(image_3)))
)