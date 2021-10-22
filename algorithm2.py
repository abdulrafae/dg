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
import torch
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--infile', '-i', type=str, help='Input file')
parser.add_argument('--outpath', '-o', type=str, help='Output path')
parser.add_argument('--seed', '-s', default=1234, type=int, help='Random seed')
args = parser.parse_args()

infile = args.infile
outpth = args.outpath
seed = args.seed

random.seed(seed)

file_exists = False
data = dict()
chars_dict = dict()
vocab = defaultdict(str)
vocab_list = []
print("Reading Files!")

infile = args.infile
with codecs.open(infile,'r',encoding='utf-8') as in_data:
	for line in tqdm(in_data.readlines()):
		words = line.strip().split(' : ')
		vocab[words[0]] = int(words[1])
			
vocab_list = list(vocab.keys())

encoded_dict = defaultdict(list)
for word in vocab.keys():
	encoded_word = jellyfish.metaphone(word)
	encoded_dict[encoded_word].append(word)

encoded_cluster_list = []
for i,key in enumerate(encoded_dict.keys()):
	encoded_cluster_list.append(len(encoded_dict[key]))

print("Metaphone DONE!")
labels, values = zip(*Counter(encoded_cluster_list).items())
bins = np.arange(len(labels)) - 0.5
entries, bin_edges, patches = plt.hist(encoded_cluster_list, bins=bins, label='Data')
bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
def fit_function(k, lamb):
	return poisson.pmf(k, lamb)
parameters, cov_matrix = curve_fit(fit_function, bin_middles, entries)

tmp_vocab = copy.deepcopy(vocab_list)
n = len(vocab_list)
K = int(0.2*n)

K = len(encoded_dict.keys())

metaphone_dict = defaultdict(list)
for word in vocab.keys():
	encoded_word = jellyfish.metaphone(word)
	if encoded_word == '':
		metaphone_dict[word].append(word)
	else:
		metaphone_dict[encoded_word].append(word)

n = len(vocab_list)
percentage = 0.2
out_clusters = int(percentage*n)
max_len = 0
max_key = ""
for key in metaphone_dict.keys():
	if max_len < len(metaphone_dict[key]):
		max_len = len(metaphone_dict[key])
		max_key = key

K = max_len
print("Max Metaphone size = "+str(K))

uniform_cluster_count = int(n/K)
uniform_clusters = defaultdict(list)
for i in range(n):
	idx = random.randint(1,uniform_cluster_count)
	uniform_clusters[idx].append(vocab_list[i])

print("Start Diverse Clusters!")
new_meta_dict = copy.deepcopy(metaphone_dict)
metaphone_cluster_count = len(metaphone_dict.keys())
tmp_vocab = copy.deepcopy(vocab_list)
final_clusters = defaultdict(list)
key_i = 0

current_count = n
for key in metaphone_dict.keys():
	if current_count == 0:
		break
	cluster = metaphone_dict[key]
	cluster_len = len(metaphone_dict[key])
	if cluster_len == K:
		cluster_list = []
		for key2 in new_meta_dict.keys():
			new_cluster = new_meta_dict[key2]
			if len(new_cluster) > 0:
				idx = random.randint(len(new_cluster))
				word = new_cluster[idx]
				cluster_list.append(word)
				del new_cluster[idx]
				new_meta_dict[key2] = new_cluster				
		final_clusters[str(key_i)] = cluster_list
		key_i += 1
		print("Adding "+str(len(cluster_list))+" elements in cluster "+str(key_i))
	elif cluster_len==0:
		exit()
	else:
		j = cluster_len
		sample_count = int(((j/float(K)))*len(list(new_meta_dict.keys())))
		if sample_count == 0:
			exit()
		selected_keys = random.sample(new_meta_dict.keys(),sample_count)
		cluster_list = []
		for key2 in selected_keys:
			new_cluster = new_meta_dict[key]
			if len(new_cluster) > 0:
				idx = random.randint(len(new_cluster))
				word = new_cluster[idx]
				cluster_list.append(new_cluster[idx])
				del new_cluster[idx]
				new_meta_dict[key2] = new_cluster
		final_clusters[str(key_i)] = cluster_list
		key_i += 1
		print("Adding "+str(len(cluster_list))+" elements in cluster "+str(key_i))
	current_count -= len(cluster_list)

print("Writing Groups!")
out_file = codecs.open(outpath+"/algo2_grouping.txt",'w',encoding='utf-8')
for key in final_clusters.keys():
	for word in final_clusters[key]:
		out_file.write(word+" : "+str(key)+"\n")
out_file.close()
print("DONE!")