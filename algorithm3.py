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
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--infile', '-i', type=str, help='Input file')
parser.add_argument('--outpath', '-o', type=str, help='Output path')
parser.add_argument('--wordvecfile', '-v', type=str, help='Word2Vec model file')
parser.add_argument('--matrixfile', '-x', type=str, help='Distance matrix file')
parser.add_argument('--seed', '-s', default=1234, type=int, help='Random seed')
args = parser.parse_args()

infile = args.infile
outpth = args.outpath
matrixfile = args.matrixfile
wordvecfile = args.wordvecfile
seed = args.seed

random.seed(seed)

infile = args.infile
wordvecfile = args.wordvecfile
print("Reading Files!")

vocab_list = list(model.wv.vocab.keys())

encoded_dict = defaultdict(list)
for word in vocab_list:
	encoded_word = jellyfish.metaphone(word)
	encoded_dict[encoded_word].append(word)

encoded_cluster_list = []
for i,key in enumerate(encoded_dict.keys()):
	encoded_cluster_list.append(len(encoded_dict[key]))

model = Word2Vec()
model.load(wordvecfile)

if matrixfile == None:
	print("Creating distance matrix!")
	dist_matrix = np.zeros((n,n))
	i = 0
	for word1 in tqdm(vocab_list):
		for j,word2 in enumerate(vocab_list):
			dist_matrix[i][j] = scipy.spatial.distance.cosine(model.wv[word1],model.wv[word2])
		i += 1
	np.save('dist_matrix.npy',dist_matrix)
	print("Distance matrix saved!")
else:
	dist_matrix = np.load(matrixfile)

n = len(vocab_list)
K = len(encoded_dict.keys())
tmp_vocab = copy.deepcopy(vocab_list)
i = 0
ranking_list = []
ranking_list_idx = []

word = random.choice(tmp_vocab)
ranking_list.append(word)
ranking_list_idx.append(tmp_vocab.index(word))
tmp_vocab.remove(word)

total_dist = 0
prev_word = word
#diverse_vocab = codecs.open(vocabfile,'w',encoding='utf-8')
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
	#diverse_vocab.write(furthest_word+"\n")
	prev_word = furthest_word
#diverse_vocab.close()

final_clusters = defaultdict(list)
for i in range(len(ranking_list)):
	cluster_idx = i%K
	final_clusters[str(cluster_idx)].append(ranking_list[i])

print("Writing Groups!")
rank_file = codecs.open(outpath+"/algo3_grouping."+lang,'w',encoding='utf-8')
for key in final_clusters.keys():
	for word in final_clusters[key]:
		rank_file.write(word+" : "+str(key)+"\n")
rank_file.close()

print("DONE!")
