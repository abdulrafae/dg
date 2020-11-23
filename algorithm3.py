from itertools import product
from string import ascii_lowercase
from gensim.models import Word2Vec
from collections import defaultdict
import codecs
from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
import math
import copy
import random
from random import shuffle
import jellyfish
from collections import Counter
import scipy
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from scipy.stats import norm
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from nltk.util import ngrams

random.seed(1234)

lang = 'txt'
data = dict()
base_path='/data/'
filenames = sys.argv[1]
emb = 100
chars_dict = dict()
vocab = defaultdict(str)
vocab_list = []
data = []
print("Reading Files!")

for filename in filenames:
	with codecs.open(filename,'r',encoding='utf-8') as in_data:
		for line in in_data:
			line = line.strip().split()
			data.append(line)
			for word in line:
				vocab[word] += 1

vocab_list = list(vocab.keys())

encoded_dict = defaultdict(list)
for word in vocab.keys():
	encoded_word = jellyfish.metaphone(word)
	encoded_dict[encoded_word].append(word)

encoded_cluster_list = []
for i,key in enumerate(encoded_dict.keys()):
	encoded_cluster_list.append(len(encoded_dict[key]))

print("Metaphone DONE!")	
K = len(encoded_dict.keys())

#model = Word2Vec(data, size=emb, window=5, min_count=1, workers=4)
#model.save(base_path+"data/w2v_models/"+lang+"_"+str(emb)+".model")
model = Word2Vec()
model.load(sys.argv[2])

vocab_dict = dict()
for i,word in enumerate(vocab_list):
	vocab_dict[word] = i

n = len(vocab_list)
dist_matrix = np.zeros((n,n))
i=0	

for word1 in tqdm(vocab_list):
	for j,word2 in enumerate(vocab_list):
		dist_matrix[i][j] = scipy.spatial.distance.cosine(model.wv[word1],model.wv[word2])
	i += 1
np.save('dist_matrix.npy',dist_matrix)
print("Matrix saved!")

#dist_matrix = np.load('dist_matrix.npy')

n = len(vocab_list)
K = len(encoded_dict.keys())
tmp_vocab = copy.deepcopy(vocab_list)
ranking_dict = dict()
i = 0
word = random.choice(tmp_vocab)
ranking_list = []
ranking_list_idx = []
ranking_list.append(word)
ranking_list_idx.append(tmp_vocab.index(word))
tmp_vocab.remove(word)
x = 0

total_dist = 0
prev_word = word
diverse_vocab = codecs.open("data/mapping/"+lang+"-"+tgt+"/randomrank_vocab."+lang,'w',encoding='utf-8')
for x in tqdm(range(n)):
	furthest_word = word
	max_dist = 0
	closest_to_i_word = []
	closest_to_i_dist = []
	tmp_idx = []
	for word2_idx,word2 in enumerate(tmp_vocab):
		submatrix = dist_matrix[tmp_idx,:][:,ranking_list_idx]
		col_idx = np.argmin(submatrix,axis=1)
		closest_idx = submatrix[np.arange(len(min_idx)),min_idx]
		tmp_vocab[tmp_idx]
		min_idx = np.argmin(map(dist_matrix[word2_idx].__getitem__, ranking_list_idx))
		closest_word = ranking_list[min_idx]
		min_dist = dist_matrix[word2_idx][min_idx]
		if closest_word == "":
			print("Closest not found!")
			exit()
		closest_to_i_word.append(closest_word)
		closest_to_i_dist.append(min_dist)
	if len(closest_to_i_dist) != len(tmp_vocab):
		print("Closest not same as vocab!")
		exit()
	idx = closest_to_i_dist.index(max(closest_to_i_dist))
	furthest_word = tmp_vocab[idx]
	ranking_list.append(furthest_word)
	ranking_list_idx.append(tmp_vocab.index(furthest_word))
	tmp_vocab.remove(furthest_word)
	diverse_vocab.write(furthest_word+"\n")
	prev_word = furthest_word

diverse_vocab.close()
final_clusters = defaultdict(list)
for i in range(len(ranking_list)):
	cluster_idx = i%K
	final_clusters[str(cluster_idx)].append(ranking_list[i])

print("Writing Groups!")
rank_file = codecs.open("data/mapping/algo3_mapping."+lang,'w',encoding='utf-8')
for key in final_clusters.keys():
	for word in final_clusters[key]:
		rank_file.write(word+" : "+str(key)+"\n")
rank_file.close()

print("DONE!")
