import array
import gzip
import json
import os
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from gtda.homology import VietorisRipsPersistence
import gudhi as gd
import gudhi.representations
from gensim.models import KeyedVectors
import re

np.random.seed(123)

folder = './Musical_Instruments_text/'
name = 'Musical_Instruments'
bert_path = './sentence-bert/stsb-roberta-large/'
bert_model = SentenceTransformer(bert_path)
core = 5

if not os.path.exists(folder + '%d-core'%core):
    os.makedirs(folder + '%d-core'%core)


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

print("----------parse metadata----------")
if not os.path.exists(folder + "meta-data/meta.json"):
    with open(folder + "meta-data/meta.json", 'w') as f:
        for l in parse(folder + 'meta-data/' + "meta_%s.json.gz"%(name)):
            f.write(l+'\n')

print("----------parse data----------")
if not os.path.exists(folder + "meta-data/%d-core.json" % core):
    with open(folder + "meta-data/%d-core.json" % core, 'w') as f:
        for l in parse(folder + 'meta-data/' + "reviews_%s_%d.json.gz"%(name, core)):
            f.write(l+'\n')

print("----------load data----------")
jsons = []
for line in open(folder + "meta-data/%d-core.json" % core).readlines():
    jsons.append(json.loads(line))


print("----------Build dict----------")
items = set()
users = set()
for j in jsons:
    items.add(j['asin'])
    users.add(j['reviewerID'])
print("n_items:", len(items), "n_users:", len(users))


item2id = {}
with open(folder + '%d-core/item_list.txt'%core, 'w') as f:
    for i, item in enumerate(items):
        item2id[item] = i
        f.writelines(item+'\t'+str(i)+'\n')

user2id =  {}
with open(folder + '%d-core/user_list.txt'%core, 'w') as f:
    for i, user in enumerate(users):
        user2id[user] = i
        f.writelines(user+'\t'+str(i)+'\n')


ui = defaultdict(list)
for j in jsons:
    u_id = user2id[j['reviewerID']]
    i_id = item2id[j['asin']]
    ui[u_id].append(i_id)
with open(folder + '%d-core/user-item-dict.json'%core, 'w') as f:
    f.write(json.dumps(ui))


print("----------Split Data----------")
train_json = {}
val_json = {}
test_json = {}
for u, items in ui.items():
    if len(items) < 10:
        testval = np.random.choice(len(items), 2, replace=False)
    else:
        testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

    test = testval[:len(testval)//2]
    val = testval[len(testval)//2:]
    train = [i for i in list(range(len(items))) if i not in testval]
    train_json[u] = [items[idx] for idx in train]
    val_json[u] = [items[idx] for idx in val.tolist()]
    test_json[u] = [items[idx] for idx in test.tolist()]

with open(folder + '%d-core/train.json'%core, 'w') as f:
    json.dump(train_json, f)
with open(folder + '%d-core/val.json'%core, 'w') as f:
    json.dump(val_json, f)
with open(folder + '%d-core/test.json'%core, 'w') as f:
    json.dump(test_json, f)


jsons = []
with open(folder + "meta-data/meta.json", 'r') as f:
    for line in f.readlines():
        jsons.append(json.loads(line))

print("----------Text Features----------")
raw_text = {}

for json in jsons:
    if json['asin'] in item2id:
        string = ' '
        if 'categories' in json:
            for cates in json['categories']:
                for cate in cates:
                    string += cate + ' '
        if 'title' in json:
            string += json['title']
        if 'brand' in json:
            # hauria de ser brand??
            string += json['brand']
        if 'description' in json:
            string += json['description']
        raw_text[item2id[json['asin']]] = string.replace('\n', ' ')
texts = []

google_vectors_2 = KeyedVectors.load('google.d2v')

def compute_TDA(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z']", " ", sentence)
    words = list(set([w for w in sentence.split() if w in google_vectors_2.key_to_index]))
    n = len(words)

    print(sentence)

    dissimilarity = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            s = google_vectors_2.similarity(words[i], words[j])
            dissimilarity[i][j] = 1 - s
            dissimilarity[j][i] = 1 - s

    # Instantiate topological transformer
    VR = VietorisRipsPersistence(metric="precomputed")

    # Compute persistence diagrams corresponding to each entry (only one here) in X
    persistence = VR.fit_transform([dissimilarity])[0]

    # Descriptors
    persistence_0 = persistence[persistence[:, -1] == 0][:, :2]
    persistence_1 = persistence[persistence[:, -1] == 1][:, :2]
    persistence_0_no_inf = np.array([bars for bars in persistence_0 if bars[1] != np.inf])
    persistence_1_no_inf = np.array([bars for bars in persistence_1 if bars[1] != np.inf])

    # Persistencia total
    pt_0 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_0_no_inf), dtype=np.dtype(np.float64)))
    pt_1 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_1_no_inf), dtype=np.dtype(np.float64)))

    # Vida mitja
    al_0 = 0
    al_1 = 0

    # Desviacio estandard
    sd_0 = 0
    sd_1 = 0

    # Entropia
    PE = gd.representations.Entropy()
    pe_0 = 0
    pe_1 = 0

    # Betti numbers
    bc = gd.representations.vector_methods.BettiCurve()
    bc_0 = np.zeros(100)
    bc_1 = np.zeros(100)

    # Landscapes
    num_landscapes = 5
    points_per_landscape = 50
    lc = gd.representations.Landscape(num_landscapes=num_landscapes, resolution=points_per_landscape)
    area_under_lc_0 = np.zeros(num_landscapes)
    area_under_lc_1 = np.zeros(num_landscapes)

    # Silhouettes
    p = 2
    resolution = 50
    s = gd.representations.Silhouette()
    s2 = gd.representations.Silhouette(weight=lambda x: np.power(x[1] - x[0], p), resolution=resolution)
    area_under_s_0 = 0
    area_under_s_1 = 0
    area_under_s2_0 = 0
    area_under_s2_1 = 0

    if (persistence_0_no_inf.size > 0):
        al_0 = pt_0 / len(persistence_0_no_inf)
        sd_0 = np.std([(start + end) / 2 for start, end in persistence_0_no_inf])
        pe_0 = PE.fit_transform([persistence_0_no_inf])[0][0]
        bc_0 = bc(persistence_0_no_inf)
        reshaped_landscapes_0 = lc(persistence_0_no_inf).reshape(num_landscapes,points_per_landscape)
        for i in range(num_landscapes):
            area_under_lc_0[i] = np.trapz(reshaped_landscapes_0[i], dx=1)
        s_0 = s(persistence_0_no_inf)
        s2_0 = s2(persistence_0_no_inf)
        area_under_s_0 = np.trapz(s_0, dx=1)
        area_under_s2_0 = np.trapz(s2_0, dx=1)

    if (persistence_1_no_inf.size > 0):
        al_1 = pt_1 / len(persistence_1_no_inf)
        sd_1 = np.std([(start + end) / 2 for start, end in persistence_1_no_inf])
        pe_1 = PE.fit_transform([persistence_1_no_inf])[0][0]
        bc_1 = bc(persistence_1_no_inf)
        reshaped_landscapes_1 = lc(persistence_1_no_inf).reshape(num_landscapes, points_per_landscape)
        for i in range(num_landscapes):
            area_under_lc_1[i] = np.trapz(reshaped_landscapes_1[i], dx=1)
        s_1 = s(persistence_1_no_inf)
        s2_1 = s2(persistence_1_no_inf)
        area_under_s_1 = np.trapz(s_1, dx=1)
        area_under_s2_1 = np.trapz(s2_1, dx=1)

    # Afegir descriptor
    return np.nan_to_num(np.concatenate(
        (np.array([pt_0, pt_1, al_0, al_1, sd_0, sd_1, pe_0, pe_1, area_under_s_0, area_under_s_1, area_under_s2_0,
                   area_under_s2_1]), area_under_lc_0, area_under_lc_1, np.array(bc_0), np.array(bc_1))), nan=0)

# bert

with open(folder + '%d-core/raw_text.txt'%core, 'w') as f:
    for i in range(len(item2id)):
        f.write(raw_text[i] + '\n')
        texts.append(raw_text[i] + '\n')
sentence_embeddings = bert_model.encode(texts)

complete_embeddings = []

#TDA i concat
for i in range(len(texts)):
    original_values = sentence_embeddings[i]
    tda = compute_TDA(texts[i])
    complete_embeddings.append(np.concatenate((original_values, tda), axis=0))

complete_embeddings = np.array(complete_embeddings)

assert complete_embeddings.shape[0] == len(item2id)

np.save(folder+'text_feat.npy', np.array(complete_embeddings))


print("----------Image Features----------")
def readImageFeatures(path):
    f = open(path, 'rb')
    while True:
        asin = f.read(10).decode('UTF-8')
        if asin == '': break
        a = array.array('f')
        a.fromfile(f, 4096)
        yield asin, a.tolist()

data = readImageFeatures(folder + 'meta-data/' + "image_features_%s.b" % name)
feats = {}
avg = []
for d in data:
    if d[0] in item2id:
        feats[int(item2id[d[0]])] = d[1]
        avg.append(d[1]) 
avg = np.array(avg).mean(0).tolist()

ret = []
for i in range(len(item2id)):
    if i in feats:
        ret.append(feats[i])
    else:
        ret.append(avg)

assert len(ret) == len(item2id)
np.save(folder+'image_feat.npy', np.array(ret))


