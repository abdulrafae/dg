import codecs
from gensim.models import Word2Vec
from collections import defaultdict
import codecs
from pathlib import Path
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.cluster import KMeans
import jellyfish
import random
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--infile', '-i', type=str, help='Input file')
parser.add_argument('--outpath', '-o', type=str, help='Output path')
args = parser.parse_args()

infile = args.infile
outpth = args.outpath

vocab_py_to_char = defaultdict(set)
vocab_char_to_py = defaultdict(set)

vocab = defaultdict(int)
with codecs.open(infile,'r',encoding='utf-8') as in_data:
	for line in tqdm(in_data.readlines()):
		words = line.strip().split(' : ')
		vocab[words[0]] = int(words[1])
	
for word in vocab.keys():
	py = jellyfish.metaphone(word)
	if py == '':
		py = word
	vocab_py_to_char[py].add(word)
	vocab_char_to_py[word].add(py)
	
K = len(vocab_py_to_char.keys())
	
tmp = copy.deepcopy(list(vocab))	

random_mapping = dict()
lengths = []
for key in vocab_py_to_char.keys():
	size = len(vocab_py_to_char[key])
	lengths.append(size)

id = 0
while len(tmp)!=0:
	size = random.choice(lengths)
	while size==0:
		size = random.choice(lengths)
	for i in range(size):
		word = random.choice(tmp)
		random_mapping[word] = id
		tmp.remove(word)
	lengths.remove(size)
	id += 1

target = codecs.open(outpath+'/algo1_grouping.txt','w',encoding='utf-8')
for key in random_mapping.keys():
	target.write(str(key)+" : "+str(random_mapping[key])+"\n")
target.close()


