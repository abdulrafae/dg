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

id = 0
while len(tmp)!=0:
	size = np.random.poisson(lam=parameters[0])
	while size==0:	
		size = np.random.poisson(lam=parameters[0])
	for i in range(size):
		word = random.choice(tmp)
		random_mapping[word] = id
		tmp.remove(word)
	lengths.remove(size)
	id += 1

target = codecs.open(outpath+'/algo2_grouping.txt','w',encoding='utf-8')
for key in random_mapping.keys():
	target.write(str(key)+" : "+str(random_mapping[key])+"\n")
target.close()
print("DONE!")
