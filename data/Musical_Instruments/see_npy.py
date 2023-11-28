import numpy as np
images = np.load('image_feat.npy')
print(images.shape)

text = np.load('text_feat.npy')
print(text.shape)

import array

'''
def readImageFeatures(path):
    f = open(path, 'rb')
    for i in range(5):
        asin = f.read(10).decode('UTF-8')
        if asin == '': break
        a = array.array('f')
        a.fromfile(f, 4096)
        yield asin, a.tolist()

data = readImageFeatures("meta-data/image_features_Musical_Instruments.b")
for d in data:
    print(d)
    print(d[0], d[1]) 
    
print('patata')
'''
