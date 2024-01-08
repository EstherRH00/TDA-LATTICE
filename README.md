# SET UP process
0. Download **5-core reviews data**, **meta data**, **image features** and **ratings-only** from [Amazon product dataset](http://jmcauley.ucsd.edu/data/amazon/links.html). Put data into the directory `dataset/meta-data/`.
1. Install [sentence-transformers](https://www.sbert.net/docs/installation.html) and download [pretrained models](https://www.sbert.net/docs/pretrained_models.html) to extract textual features. You can do so by running
```
   cd data/sentence-transformers
   git lfs install
   git clone https://huggingface.co/sentence-transformers/stsb-roberta-large
```
2. Install word2vec: Simply run `data/text_aux.py`

It should look like this:
```
  ├─ data/: 
      ├── Baby/
      	├── meta-data/
      		├── image_features_Baby.b
      		├── meta-Baby.json.gz
      		├── reviews_Baby_5.json.gz
      		├── ratings_Baby.csv
      ├── sentence-transformers/
          	├── stsb-roberta-large
      ├── google.d2v
      ├── google.d2v.vectors.npy
```

3. Build the data; simply run the file `data/build_data_TDA.py`

## Run models
To run **lattice**, **lattice_tda_first_graph**, **lattice_tda_each_graph**, **lattice_tda_drop_nodes**, **mf**, **ngcf** or **lightgcn**; run:
```
cd codes
python main.py --model lattice --dataset Baby
```
This will run the default data built; if you want to use topological data for text and/or image, add `--textTDA True` and/or `--imageTDA True`; for example:
`python main.py --model lattice --dataset Baby --textTDA True` will run with the default data for image and with topological data for text.

If you are using lattice_tda_drop_nodes, you can set the percentatge of nodes dropped by using `--percentNodesDropped`. It is defaulty set at 25. The following code would drop the 1% of the nodes that affect the graph less, topologically speaking: `python main.py --model lattice_tda_drop_nodes --dataset Baby --percentNodesDropped 1`

To run **vbpr**, **grcn** or **mmgcn**, run:
```
cd MMRecModels
python main.py --model VBPR --dataset Baby
```

## Acknowledgement

The structure of this code is largely based on [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), [MICRO](https://github.com/CRIPAC-DIG/MICRO/) and [MMRec](https://github.com/enoche/MMRec/tree/master). Thank for their work.