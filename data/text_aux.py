import gensim.downloader

google_vectors = gensim.downloader.load('word2vec-google-news-300')
google_vectors.save('google.d2v')