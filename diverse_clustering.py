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

random.seed(1234)

file_exists = False
lang = 'en'
tgt = 'de'
direction = lang+'-'+tgt#+'_old'
data = dict()
base_path='/data2/Diverse_Clustering/'
#dataset='KFFT_ja_en/'
path = base_path+'/data/'
filenames = [path+'vocab.news.'+direction+'.'+lang]#'UNv1.0.ar-zh.zh'
emb = 100
chars_dict = dict()
vocab = defaultdict(str)
vocab_list = []
data = []
print("Reading Files!")

train_vocab = defaultdict(int)
valid_vocab = defaultdict(int)
test_vocab = defaultdict(int)

for filename in filenames:
	with codecs.open(filename,'r',encoding='utf-8') as in_data:
		for line in tqdm(in_data.readlines()):
			words = line.strip().split(' : ')
			vocab[words[0]] = words[1]
vocab_list = list(vocab.keys())

encoded_dict = defaultdict(list)
for word in vocab.keys():
	encoded_word = jellyfish.metaphone(word)
	encoded_dict[encoded_word].append(word)

encoded_cluster_list = []
for i,key in enumerate(encoded_dict.keys()):
	encoded_cluster_list.append(len(encoded_dict[key]))

print("Metaphone DONE!")
	
def knn_random(points,vocab_list):
	
	print("START KNN Clustering!")
	n = len(vocab_list)
	percentage = 0.2
	percentage_rev = int(1/percentage)
	K = percentage*n
	
	nbrs = NearestNeighbors(n_neighbors=percentage_rev, algorithm='ball_tree').fit(points)
	distances, indices = nbrs.kneighbors(points)
	knn_clusters = defaultdict(set)
	for i in range(indices.shape[0]):
		for j in range(1,indices.shape[1]):
			knn_clusters[i].add(indices[i,j])
	#print(knn_clusters.items()[:10])
	knn_clusters = list(knn_clusters)
	
	knn_file = open('knnclusters.out','w')
	for i in range(indices.shape[0]):
		for j in range(indices.shape[1]):
			knn_file.write(str(knn_clusters[i][j])+ " ")
		knn_file.write('\n')
	knn_file.close()
	
	knn_max_len = 0
	knn_max_cluster = -1
	for key in knn_clusters.keys():
		if knn_max_len < len(knn_clusters[key]):
			knn_max_len = len(knn_clusters[key])
			knn_max_cluster = key
			
	print("START Diverse Clustering!")
	random_clusters = defaultdict(list)
	j = 0
	i=0
	while j < knn_max_len:
		for knn_cluster_idx in knn_clusters.keys():
			knn_cluster	= knn_clusters[knn_cluster_idx]
			if len(knn_cluster) > j:
				word = knn_cluster[j]
				random_clusters[i].append(word)
		j += 1
		i += 1

	print("Writing KNN-Clustering Files!")
	kmeans_file = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/knn_mapping."+lang,'w',encoding='utf-8')
	for key in knn_cluster_dict.keys():
		for word in knn_cluster_dict[key]:
			kmeans_file.write(word+","+str(key)+"\n")
	kmeans_file.close()

	random_file = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/randomknn_mapping."+lang,'w',encoding='utf-8')
	for key in random_clusters.keys():
		for word in random_clusters[key]:
			random_file.write(word+","+str(key)+"\n")
	random_file.close()
	
	#exit()	
	
	
def uniform_random(points,vocab_list):
	
	print("START Uniform Clustering!")
	n = len(vocab_list)
	percentage = 0.2
	percentage_rev = int(1/percentage)
	K = percentage*n

	uniform_clusters = defaultdict(list)
	for i in range(n):
		idx = random.randint(1,percentage_rev)
		uniform_clusters[idx].append(vocab_list[i])

	uniform_max_len = 0
	uniform_max_cluster = -1
	for key in uniform_clusters.keys():
		if uniform_max_len < len(uniform_clusters[key]):
			uniform_max_len = len(uniform_clusters[key])
			uniform_max_cluster = key
			
	print("START Uniform-Diverse Clustering!")
	print(uniform_max_len)
	random_clusters = defaultdict(list)
	j = 0
	i=0
	while j < uniform_max_len:
		for uniform_cluster_idx in uniform_clusters.keys():
			uniform_cluster	= uniform_clusters[uniform_cluster_idx]
			if len(uniform_cluster) > j:
				word = uniform_cluster[j]
				random_clusters[i].append(word)
		j += 1
		i += 1
			
	#print("Writing Uniform-Clustering Files!")
	#kmeans_file = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/uniform_mapping."+lang,'w',encoding='utf-8')
	#for key in uniform_clusters.keys():
	#	for word in uniform_clusters[key]:
	#		kmeans_file.write(word+","+str(key)+"\n")
	#kmeans_file.close()

	#random_file = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/randomuni_mapping."+lang,'w',encoding='utf-8')
	#for key in random_clusters.keys():
	#	for word in random_clusters[key]:
	#		random_file.write(word+","+str(key)+"\n")
	#random_file.close()

	total = 0
	for key in uniform_clusters.keys():
		val = len(uniform_clusters[key])
		total += val
	avg = float(total)/len(uniform_clusters.keys())
	print("Avg Uniform-Cluster length "+str(avg))
	print("No of Uniform-Cluster "+str(len(uniform_clusters.keys())))
	
	total = 0
	for key in random_clusters.keys():
		val = len(random_clusters[key])
		total += val
	avg = float(total)/len(random_clusters.keys())
	print("Avg RandUniform-Cluster length "+str(avg))
	print("No of RandUniform-Cluster "+str(len(random_clusters.keys())))
	
	#max_len = 0
	#min_len = n+1
	#max_cluster = -1
	#min_cluster = -1
	#for key in random_clusters.keys():
	#	if max_len < len(random_clusters[key]):
	#		max_len = len(random_clusters[key])
	#		max_cluster = key
	#	if min_len > len(random_clusters[key]):
	#		min_len = len(random_clusters[key])
	#		min_cluster = key
	#print(list(random_clusters.items())[:2])
	#print(" Max len is "+str(max_len)+" of cluster "+str(max_cluster))
	#print(" Min len is "+str(min_len)+" of cluster "+str(min_cluster))
	
def fit_curve_random(vocab_list,encoded_cluster_list):	
	print("START CurveFit!")
	#labels, values = zip(*Counter(encoded_cluster_list).items())
	#indexes = np.arange(len(labels))
	#width = 1
	#plt.bar(indexes, values, width)	
		
	#plt.hist(encoded_cluster_list, width=0.2, color='b')	

	#bins = np.arange(len(labels)) - 0.5
	#entries, bin_edges, patches = plt.hist(encoded_cluster_list, bins=bins, label='Data')
	#plt.show()
	# calculate bin centres
	#bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
	
	#CHECKING POISSON
	#def fit_function(k, lamb):
	#	'''poisson function, parameter lamb is the fit parameter'''
	#	return poisson.pmf(k, lamb)
	#parameters, cov_matrix = curve_fit(fit_function, bin_middles, entries)
	#print(parameters)
	parameters = [1.0954711]
	#RESULT = 1.0954711
	tmp_vocab = copy.deepcopy(vocab_list)
	n = len(vocab_list)
	K = int(0.2*n)
	poisson_clusters = defaultdict(list)
	total_size = 0
	idx = 0
	#for i in range(K): 
	while len(tmp_vocab) > 0:
		print(idx)
		sampled_len = np.random.poisson(parameters[0])
		while sampled_len==0:
			sampled_len = np.random.poisson(parameters[0])
		if len(tmp_vocab)<sampled_len:
			sampled_len = len(tmp_vocab)
		for j in range(sampled_len):
			word = random.choice(tmp_vocab)
			poisson_clusters[idx].append(word)
			tmp_vocab.remove(word)
		idx += 1
	#print(parameters)
	#print(np.sqrt(np.diag(cov_matrix))[0])
	#poisson_mse = mse(encoded_cluster_list)
	print("Writing CurveFit-Clustering Files!")
	poisson_file = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/randompoisson_mapping."+lang,'w',encoding='utf-8')
	for key in poisson_clusters.keys():
		for word in poisson_clusters[key]:
			poisson_file.write(word+","+str(key)+"\n")
	poisson_file.close()

	#CHECKING GAUSSIAN
	#def gauss_function(x, a, x0, sigma):
	#	return a*np.exp(-(x-x0)**2/(2*sigma**2))
	#mu, std = norm.fit(data)
	#mean = sum(bin_middles * entries)
	#sigma = sum(entries * (bin_middles - mean)**2)
	#popt, pcov = curve_fit(gauss_function, bin_middles, entries, p0 = [1, mean, sigma])
	#(1.732527460693517, 8.10419981287411)
	#print(mu)
	#print(np.sqrt(np.diag(pcov))[0])
	#exit()

	#max_len = 0
	#min_len = n+1
	#max_cluster = -1
	#min_cluster = -1
	#for key in poisson_clusters.keys():
	#	if max_len < len(poisson_clusters[key]):
	#		max_len = len(poisson_clusters[key])
	#		max_cluster = key
	#	if min_len > len(poisson_clusters[key]):
	#		min_len = len(poisson_clusters[key])
	#		min_cluster = key
	#print(list(poisson_clusters.items())[:2])
	#print("Number of clusters "+str(len(poisson_clusters.keys())))
	#print("Max len is "+str(max_len)+" of cluster "+str(max_cluster))
	#print("Min len is "+str(min_len)+" of cluster "+str(min_cluster))
	total = 0
	for key in poisson_clusters.keys():
		val = len(poisson_clusters[key])
		total += val
	avg = float(total)/len(poisson_clusters.keys())
	print("Avg Poisson-Cluster length "+str(avg))
	print("No of Poisson-Cluster "+str(len(poisson_clusters.keys())))
	
	
def max_k_method(vocab_list,metaphone_dict):
	'''
	biggest cluster is K
	n/K points
	for K size cluster	(forming one cluster)
		each from priliminary cluster
	if j<K size cluster:
		sample a j/K %age cluster from primilinary cluster
	K=10
	j=5
	sample 1/2 of clusters
	(j/K)*(n/K)
	'''
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
	for key in tqdm(metaphone_dict.keys()):
		cluster = metaphone_dict[key]
		cluster_len = len(metaphone_dict[key])
		if cluster_len == K:
			cluster_list = []
			for key2 in new_meta_dict.keys():
				new_cluster = new_meta_dict[key2]
				if len(new_cluster) > 0:
					cluster_list.append(new_cluster[0])
					del new_cluster[0]
					new_meta_dict[key] = new_cluster				
			#key_i = sample_count*(float(n)/K)
			final_clusters[str(key_i)] = cluster_list
			key_i += 1
		else:
			j = cluster_len
			sample_count = int((j*1.0)/float(K))
			selected_keys = random.sample(new_meta_dict.keys(),sample_count)
			cluster_list = []
			for key2 in selected_keys:
				new_cluster = new_meta_dict[key]
				if len(new_cluster) > 0:
					cluster_list.append(new_cluster[0])
					del new_cluster[0]
					new_meta_dict[key] = new_cluster
			#key_i = sample_count*(float(n)/K)
			final_clusters[str(key_i)] = cluster_list
			key_i += 1
		
	
	print("Writing Sample-Clustering Files!")
	sample_file = codecs.open(base_path+"/data/randomsample_mapping."+lang,'w',encoding='utf-8')
	for key in final_clusters.keys():
		for word in final_clusters[key]:
			sample_file.write(word+","+str(key)+"\n")
	sample_file.close()
	
	print("DONE!")
	#exit()
	
def rank_word_method(vocab_list,model,metaphone_dict,dist_matrix,vocab_dict):

	n = len(vocab_list)
	K = len(metaphone_dict.keys())
	tmp_vocab = copy.deepcopy(vocab_list)
	#tmp_vocab = vocab_list[:3]
	#print("Length of vocab ",len(tmp_vocab))
	ranking_dict = dict()
	i = 0
	word = random.choice(tmp_vocab)
	ranking_list = []
	ranking_list_idx = []
	ranking_list.append(word)
	ranking_list_idx.append(tmp_vocab.index(word))
	tmp_vocab.remove(word)
	x = 0
	#while len(tmp_vocab)>0:
	total_dist = 0
	prev_word = word
	diverse_vocab = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/randomrank_vocab."+lang,'w',encoding='utf-8')
	for x in tqdm(range(n)):
	#for x in tqdm(range(len(vocab_list[:2]))):
		furthest_word = word
		max_dist = 0
		closest_to_i_word = []
		closest_to_i_dist = []
		tmp_idx = []
		for word2_idx,word2 in enumerate(tmp_vocab):
			#if word2 in ranking_list:
			#	closest_to_i_word.append("")
			#	closest_to_i_dist.append(0)
			#	continue
			#tmp_idx.append(word2_idx)
			#min_dist = 1000
			#closest_word = ""
			#for word3 in ranking_list:
			#	#dist = scipy.spatial.distance.cosine(model.wv[word2],model.wv[word3])
			#	dist = dist_matrix[vocab_dict[word2]][vocab_dict[word3]]
			#	if dist < min_dist:
			#		min_dist = dist
			#		closest_word = word3
			
			submatrix = dist_matrix[tmp_idx,:][:,ranking_list_idx]
			min_idx = np.argmin(submatrix,axis=1)
			closest_idx = submatrix[np.arange(len(min_idx)),min_idx]
			#tmp_vocab[tmp_idx]
			#closest_idx = tmp_idx[col_idx]
			#row_idx = np.argmax(submatrix[min_idx])
			#tmp_idx[col_idx]
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
		#print(str(x)+"/"+str(n))
		#x += 1
	diverse_vocab.close()
	final_clusters = defaultdict(list)
	for i in range(len(ranking_list)):
		cluster_idx = i%K
		final_clusters[str(cluster_idx)].append(ranking_list[i])

	print("Writing Rank-Clustering Files!")
	rank_file = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/randomrank_mapping."+lang,'w',encoding='utf-8')
	for key in final_clusters.keys():
		for word in final_clusters[key]:
			rank_file.write(word+","+str(key)+"\n")
	rank_file.close()

	#print("DONE!")

		
def rank_word_method_new(vocab_list,model,metaphone_dict,vocab_dict):

	n = len(vocab_list)
	K = len(metaphone_dict.keys())
	tmp_vocab = copy.deepcopy(vocab_list)
	#tmp_vocab = vocab_list[:3]
	#print("Length of vocab ",len(tmp_vocab))
	ranking_dict = dict()
	i = 0
	word = random.choice(tmp_vocab)
	ranking_list = []
	ranking_list_idx = []
	ranking_list.append(word)
	ranking_list_idx.append(tmp_vocab.index(word))
	tmp_vocab.remove(word)
	x = 0
	#while len(tmp_vocab)>0:
	total_dist = 0
	prev_word = word
	diverse_vocab = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/randomrank_vocab."+lang,'w',encoding='utf-8')
	for x in tqdm(range(n)):
	#for x in tqdm(range(len(vocab_list[:2]))):
		furthest_word = word
		max_dist = 0
		closest_to_i_word = []
		closest_to_i_dist = []
		tmp_idx = []
		for word2_idx,word2 in enumerate(tmp_vocab):
			#if word2 in ranking_list:
			#	closest_to_i_word.append("")
			#	closest_to_i_dist.append(0)
			#	continue
			#tmp_idx.append(word2_idx)
			min_dist = 1000
			closest_word = ""
			for word3 in ranking_list:
				dist = scipy.spatial.distance.cosine(model.wv[word2],model.wv[word3])
				#dist = dist_matrix[vocab_dict[word2]][vocab_dict[word3]]
				if dist < min_dist:
					min_dist = dist
					closest_word = word3
			
			#submatrix = dist_matrix[tmp_idx,:][:,ranking_list_idx]
			#col_idx = np.argmin(submatrix,axis=1)
			#closest_idx = submatrix[np.arange(len(min_idx)),min_idx]
			#tmp_vocab[tmp_idx]
			#closest_idx = tmp_idx[col_idx]
			#row_idx = np.argmax(submatrix[min_idx])
			#tmp_idx[col_idx]
			#min_idx = np.argmin(map(dist_matrix[word2_idx].__getitem__, ranking_list_idx))
			#closest_word = ranking_list[min_idx]
			#min_dist = dist_matrix[word2_idx][min_idx]
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
		#print(str(x)+"/"+str(n))
		#x += 1
	diverse_vocab.close()
	final_clusters = defaultdict(list)
	for i in range(len(ranking_list)):
		cluster_idx = i%K
		final_clusters[str(cluster_idx)].append(ranking_list[i])

	print("Writing Rank-Clustering Files!")
	rank_file = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/randomrank_mapping."+lang,'w',encoding='utf-8')
	for key in final_clusters.keys():
		for word in final_clusters[key]:
			rank_file.write(word+","+str(key)+"\n")
	rank_file.close()

	#print("DONE!")
		
		
def mcmc(vocab_list,encoded_dict,data):
	#\caption{MCMC (maximize the unique bigram cluster number given the fixed number of clusters $K$}
	#\label{alg-poissoncluster}
	#\textbf{Input}: Vocabulary of bigrams $V$ with size $N$. $K$ is the biggest size of a cluster\\
	#\textbf{Parameter}:\\
	#\textbf{Output}: Assignment of each word $W$ to its cluster $C$\\
	#\begin{algorithmic}[1] %[1] enables line numbers
	#\STATE each word is uniform randomly assigned to a cluster
	#\FOR{$i<maxiteration$}
	#\FOR{ each word in each entry in the bigram vocabulary $V$}
	#\STATE sample a cluster for this word given $Pr(C|w;\theta_{-w})\propto - \sum_C P(C;\theta_{-w})\log{P(C;\theta_{-w})}$
	#\STATE update clusters for all appearance of $w$ in $V$
	#\ENDFOR
	#\ENDFOR	
	
	def get_ngram_prob(data,ngram):
		prob = defaultdict(int)
		for sentence in data:
			ngrams_list = list(ngrams(sentence.split(),ngram))
			for ngram in ngrams_list:
				ngram_str = "_".join(i for i in ngram)
				prob[ngram_str] += 1
				total_ngrams += 1
		for key in prob.keys:
			prob[key] = prob[key]/float(total_ngrams)
		return prob
	
	def get_cluster_freq(word_freq,cluster_dict):
		cluster_freq = defaultdict(int)
		for word in word_freq.keys():
			freq = word_freq[word]
			idx = cluster_dict[idx]
			cluster_freq[idx] += freq
		return cluster_freq
	
	def get_shannon_prob(max_idx,cluster_freq):
		total_count = sum(cluster_freq.values())
		cluster_probs = []
		total = 0
		for key in max_idx:
			prob = cluster_freq[key]/float(total_count)
			entropy = prob*math.log(prob)
			cluster_probs.append(prob)
		total = float(sum(cluster_probs))
		for key in range(len(cluster_probs)):
			cluster_probs[key] = cluster_probs[key]/total
		return cluster_probs

	def update_cluster(word,new_idx,old_idx,cluster_freq,word_freq):
		cluster_freq[old_idx] = cluster_freq[old_idx]-word_freq[word]
		cluster_freq[new_idx] = cluster_freq[new_idx]-word_freq[word]
		
		
	word_freq = get_ngram_prob(data,2)
	
	percentage = 0.2
	K = percentage*n
	
	uniform_cluster_count = int(n/K)
	uniform_clusters = defaultdict(list)
	uniform_clusters_words = defaultdict(int)
	for i in range(n):
		idx = random.randint(1,uniform_cluster_count)
		uniform_clusters[idx].append(vocab_list[i])
		uniform_clusters_words[word] = idx
	
	cluster_freq = get_cluster_freq(word_freq,uniform_clusters) 
	
	cluster_prob = get_shannon_prob(min(uniform_clusters.keys()),max(uniform_clusters.keys()),cluster_freq)
	
	maxIter = 1
	iter = 0
	while iter < maxIter:
		for bigram in tqdm(bigram_prob.keys()):
			for word in bigram.split("_"):
				#$Pr(C|w;\theta_{-w})\propto - \sum_C P(C;\theta_{-w})\log{P(C;\theta_{-w})}$
				selected_cluster = random.choice(uniform_clusters,cluster_prob)
				update_cluster(word,selected_cluster,uniform_clusters_words[word],cluster_freq,word_freq)
		iter += 1

'''		
def maximize_entropy(vocab_list,encoded_dict,data):

	uniform_cluster_count = int(n/K)
	uniform_clusters = defaultdict(list)
	for i in range(n):
		idx = random.randint(1,uniform_cluster_count)
		uniform_clusters[idx].append(vocab_list[i])
	
	max_len = 0
	max_key = ""
	for key in metaphone_dict.keys():
		if max_len < len(metaphone_dict[key]):
			max_len = len(metaphone_dict[key])
			max_key = key
	K = max_len
	
	def get_shannon_prob(data,ngram):
		prob = defaultdict(int)
		for sentence in data:
			ngrams_list = list(ngrams(sentence.split(),ngram))
			for ngram in ngrams_list:
				ngram_str = "_".join(i for i in ngram)
				prob[ngram_str] += 1
				total_ngrams += 1
		entropy = defaultdict(int)
		for key in prob.keys:
			tmp_prob = prob[key]/float(total_ngrams)
			entropy[key] = tmp_prob*math.log(tmp_prob)
		return entropy

	def update_shannon_entropy(entropy,w,cluster_dict)
'''		
	
K = len(encoded_dict.keys())

#print("Model is "+lang+"_"+str(emb)+".model")
#model = Word2Vec(data, size=emb, window=5, min_count=1, workers=4)
#model.save(base_path+dataset+"/w2v_models/"+lang+"_"+str(emb)+".model")

#print("Model is "+lang+"_"+str(emb)+".model")
#model = Word2Vec.load(base_path+"/w2v_models/"+lang+"_"+str(emb)+".model")

#points = np.zeros((len(vocab_list),emb))
#for i,word in enumerate(vocab.keys()):
#	points[i,:] = model.wv[word]

vocab_dict = dict()
for i,word in enumerate(vocab_list):
	vocab_dict[word] = i

n = len(vocab_list)
#dist_matrix = np.zeros((n,n))
#dist_matrix = torch.zeros((n,n))
i=0

def get_dist(dist_matrix, chunk, vocab_list):
	for index1 in tqdm(chunk):
		word1 = vocab_list[index1]
		for index2, word2 in enumerate(vocab_list):
			dist = scipy.spatial.distance.cosine(model.wv[word1],model.wv[word2])
			dist_matrix[index1][index2] = dist
	return dist_matrix	
#import threading
#threads = list()
#idx_list = [i for i in range(n)]
#thread_count = 100
#chunk_size = int(n/thread_count)
##for word1 in tqdm(vocab_list):
#chunks = [idx_list[i:i+chunk_size] for i in range(0,n,chunk_size)]
#print('Creating threads!')
#for index, chunk in enumerate(chunks):
#	#print(f'Main    : create and start thread {index}.')
#	x = threading.Thread(target=get_dist, args=(dist_matrix, chunk, vocab_list))
#	threads.append(x)
#	x.start()
#for index, thread in enumerate(threads):
#	#print(f'Main    : before joining thread {index}.')
#	return_dist = thread.join()
#	dist_matrix = dist_matrix + return_dist
#	#thread_costs = np.add(thread_costs,return_cost)
#	#print(f'Main    : thread {index} done')
#print('Created threads!')
vocab_len = len(vocab_list)
#i = 0
#for word1 in tqdm(vocab_list):
#	for j in range(i,vocab_len):
#		word2 = vocab_list[j]
#		dist = scipy.spatial.distance.cosine(model.wv[word1],model.wv[word2])
#		dist_matrix[i][j] = dist
#		dist_matrix[j][i] = dist
#	#print(str(i)+"/"+str(n))
#	i += 1
#np.save(base_path+dataset+'/dist_matrix.npy',dist_matrix)
#print("Matrix saved!")
#dist_matrix = np.load(base_path+dataset+'/dist_matrix.npy')

#dist_matrix = np.load('dist_matrix.npy')
#dist_matrix = np.load('dist_matrix.npy')
#print("Matrix loaded!")
#uniform_random(points,vocab_list)
#knn_random(points,vocab_list)
#fit_curve_random(vocab_list,encoded_cluster_list)
#max_k_method(vocab_list,encoded_dict)
#rank_word_method(vocab_list,model,encoded_dict,dist_matrix,vocab_dict)
#exit()	
#metaphone_dict = encoded_dict
#mcmc(vocab_list,encoded_dict,data)
#####################################################################################################################
def mcmcmt(vocab_list,encoded_dict,data):
	def get_word_freq(data):
		word_freq = defaultdict(int)
		for sentence in data:
			for word in sentence:
				word_freq[word] += 1
		return word_freq

	def get_ngram_prob(data,ngram):
		prob = defaultdict(int)
		total_ngrams = 0
		for sentence in data:
			ngrams_list = list(ngrams(sentence,ngram))
			for ngram in ngrams_list:
				ngram_str = "_".join(i for i in ngram)
				prob[ngram_str] += 1
				total_ngrams += 1
		for key in prob.keys:
			prob[key] = prob[key]/float(total_ngrams)
		return prob

	def get_cluster_freq(word_freq,cluster_dict):
		cluster_freq = dict()
		for word in word_freq.keys():
			freq = word_freq[word]
			idx = cluster_dict[word]
			try:
				cluster_freq[idx] += freq
			except:
				cluster_freq[idx] = freq
		return cluster_freq

	def get_shannon_prob(min_idx,max_idx,cluster_freq):
		total_count = sum(cluster_freq.values())
		cluster_probs = []
		total = 0
		total_entropy = 0.0
		for key in range(min_idx,max_idx+1):
		#for key in range(cluster_freq)
			prob = cluster_freq[key]/float(total_count)
			entropy = prob*math.log(prob)
			cluster_probs.append(prob)
			total_entropy += entropy

		cluster_probs = np.array(cluster_probs)/float(sum(cluster_probs))
		final_entropy = -1*total_entropy
		return cluster_probs,final_entropy

	def update_cluster(word,new_idx,old_idx,cluster_freq,word_freq,uniform_clusters_words,uniform_clusters):
		#old_cluster_entropy = get_shannon_prob(min(uniform_clusters.keys()),max(uniform_clusters.keys()),cluster_freq)
		cluster_freq[old_idx] = cluster_freq[old_idx]-word_freq[word]
		cluster_freq[new_idx] = cluster_freq[new_idx]+word_freq[word]
		#new_cluster_entropy = get_shannon_prob(min(uniform_clusters.keys()),max(uniform_clusters.keys()),cluster_freq)
		#if new_cluster_entropy<old_cluster_entropy:
		#	cluster_freq[old_idx] = cluster_freq[old_idx]+word_freq[word]
		#	cluster_freq[new_idx] = cluster_freq[new_idx]-word_freq[word]
		#	return
		uniform_clusters[old_idx].remove(word)
		uniform_clusters[new_idx].append(word)
		uniform_clusters_words[word] = new_idx
		
	def get_separate_entorpy(train_cluster_freq,valid_cluster_freq,test_cluster_freq):

		train_cluster_probs = np.array(list(train_cluster_freq.values()))/sum(train_cluster_freq.values())		
		total_entropy = 0.0
		for prob in train_cluster_probs:
			if prob > 0.0:
				total_entropy += prob*math.log(prob)
		train_entropy = -1*total_entropy

		valid_cluster_probs = np.array(list(valid_cluster_freq.values()))/sum(valid_cluster_freq.values())		
		total_entropy = 0.0
		for prob in valid_cluster_probs:
			if prob > 0.0:
				total_entropy += prob*math.log(prob)
		valid_entropy = -1*total_entropy	

		test_cluster_probs = np.array(list(test_cluster_freq.values()))/sum(test_cluster_freq.values())		
		total_entropy = 0.0
		for prob in test_cluster_probs:
			if prob > 0.0:
				total_entropy += prob*math.log(prob)
		test_entropy = -1*total_entropy	

		#print(train_entropy)
		#print(valid_entropy)
		#print(test_entropy)
		return train_entropy, valid_entropy, test_entropy
		
	def get_cluster_entropy(cluster_freq,word_freq,uniform_clusters_words):
		cluster_entropy = [0.0]*len(cluster_freq.keys())
		total_count = sum(cluster_freq.values())
		for word in word_freq.keys():
			idx = uniform_clusters_words[word]
			p = word_freq[word]/float(cluster_freq[idx])
			if p > 0.0:
				cluster_entropy[idx] += p*math.log(p)
		for idx in range(len(cluster_entropy)):
			if cluster_entropy[idx] != 0.0:
				cluster_entropy[idx] = -1*np.array(cluster_entropy[idx])
		return cluster_entropy
		
	word_freq = get_word_freq(data) 
	#bigram_prob = get_ngram_prob(data,2)

	ngram_count = 2
	bigram_prob = defaultdict(int)
	total_ngrams = 0
	for sentence in data:
		ngrams_list = list(ngrams(sentence,ngram_count))
		for ngram in ngrams_list:
			ngram_str = "_".join(i for i in ngram)
			bigram_prob[ngram_str] += 1
			total_ngrams += 1
	for key in bigram_prob.keys():
		bigram_prob[key] = bigram_prob[key]/float(total_ngrams)

		
	metaphone_dict = encoded_dict
	#max_len = 0
	#max_key = ""
	#for key in metaphone_dict.keys():
	#	if max_len < len(metaphone_dict[key]):
	#		max_len = len(metaphone_dict[key])
	#		max_key = key
		
	K = len(metaphone_dict.keys())

	uniform_cluster_count = int(K)
	uniform_clusters = defaultdict(list)
	uniform_clusters_words = defaultdict(int)
	tmp_vocab = copy.deepcopy(vocab_list)
	for i in range(n):
		idx = i%uniform_cluster_count #random.randint(0,uniform_cluster_count-1)
		word = random.choice(tmp_vocab)
		uniform_clusters[idx].append(word)
		uniform_clusters_words[word] = idx
		tmp_vocab.remove(word)

	cluster_freq = get_cluster_freq(word_freq,uniform_clusters_words) 

	#cluster_prob = get_shannon_prob(min(uniform_clusters.keys()),max(uniform_clusters.keys()),cluster_freq)
	min_idx = min(uniform_clusters.keys())
	max_idx = max(uniform_clusters.keys())
	total_count = sum(cluster_freq.values())

	#cluster_probs = []
	#total = 0
	#for key in range(min_idx,max_idx+1):
	#	prob = cluster_freq[key]/float(total_count)
	#	entropy = prob*math.log(prob)
	#	cluster_probs.append(prob)
	#total = float(sum(cluster_probs))
	#for key in range(len(cluster_probs)):
	#	cluster_probs[key] = cluster_probs[key]/total
	min_cluster_idx = min(uniform_clusters.keys())
	max_cluster_idx = max(uniform_clusters.keys())
	#cluster_prob,entropy = get_shannon_prob(min_cluster_idx,max_cluster_idx,cluster_freq)

	total_count = sum(cluster_freq.values())
	cluster_prob = []
	total = 0
	total_entropy = 0.0
	cluster_prob = np.zeros((max_cluster_idx+1))
	#for key in range(min_idx,max_idx+1):
	for key in cluster_freq.keys():
		prob = cluster_freq[key]/float(total_count)
		if prob == 0.0:
			entropy = 0.0
		else:
			entropy = prob*math.log(prob)
		cluster_prob[key] = prob
		total_entropy += entropy

	cluster_prob = np.array(cluster_prob)/float(sum(cluster_prob))

	train_cluster_freq = defaultdict(int)
	for word in train_vocab.keys():
		freq = train_vocab[word]
		idx = uniform_clusters_words[word]
		try:
			train_cluster_freq[idx] += freq
		except:
			train_cluster_freq[idx] = freq
	valid_cluster_freq = defaultdict(int)
	for word in valid_vocab.keys():
		freq = valid_vocab[word]
		idx = uniform_clusters_words[word]
		try:
			valid_cluster_freq[idx] += freq
		except:
			valid_cluster_freq[idx] = freq
	test_cluster_freq = defaultdict(int)
	for word in test_vocab.keys():
		freq = test_vocab[word]
		idx = uniform_clusters_words[word]
		try:
			test_cluster_freq[idx] += freq
		except:
			test_cluster_freq[idx] = freq	

	train_entropy,valid_entropy,test_entropy = get_separate_entorpy(train_cluster_freq,valid_cluster_freq,test_cluster_freq)

	target_train = open('Entropy_2gram_10k_train.txt','w')
	target_valid = open('Entropy_2gram_10k_valid.txt','w')
	target_test = open('Entropy_2gram_10k_test.txt','w')
	target_train.write(str(train_entropy)+"\n")
	target_valid.write(str(valid_entropy)+"\n")
	target_test.write(str(test_entropy)+"\n")

	clusters = np.zeros(max_cluster_idx+1)
	for i in range(max_cluster_idx+1):
		clusters[i] = int(i)
	#clusters = np.array(list(uniform_clusters.keys()))
	cluster_entropy = get_cluster_entropy(cluster_freq,word_freq,uniform_clusters_words)
	cluster_entropy_prob = cluster_entropy/sum(cluster_entropy)

	min_idx = np.argmin(cluster_entropy_prob)
	if cluster_entropy_prob[min_idx] < 0.0:
		print("Cluster: "+str(cluster_freq[min_idx]))
		for word in word_freq.keys():
			freq = word_freq[word]
			idx = cluster_dict[word]
			if idx == min_idx:
				print(word+": "+str(freq))
	else:
		print("No negative value!")

	maxIter = 10000
	iter = 0
	while iter < maxIter:
		for bigram in tqdm(bigram_prob.keys()):
			for word in bigram.split("_"):
				#$Pr(C|w;\theta_{-w})\propto - \sum_C P(C;\theta_{-w})\log{P(C;\theta_{-w})}$
				new_idx = int(np.random.choice(clusters,p=cluster_entropy_prob))
				old_idx = uniform_clusters_words[word]
				#old_idx = 999
				if old_idx == new_idx:
					continue

				uniform_clusters_words[word] = new_idx
				
				try:
					train_cluster_freq[old_idx] = train_cluster_freq[old_idx]-train_vocab[word]
					train_cluster_freq[new_idx] = train_cluster_freq[new_idx]+train_vocab[word]
				except:
					pass
				
				try:
					valid_cluster_freq[old_idx] = valid_cluster_freq[old_idx]-valid_vocab[word]
					valid_cluster_freq[new_idx] = valid_cluster_freq[new_idx]+valid_vocab[word]
				except:
					pass
				
				try:
					test_cluster_freq[old_idx] = test_cluster_freq[old_idx]-test_vocab[word]
					test_cluster_freq[new_idx] = test_cluster_freq[new_idx]+test_vocab[word]
				except:
					pass
					
				print("1 iteration almost done!")
				#cluster_prob = np.array(list(cluster_freq.values()))/float(sum(cluster_freq.values()))
				cluster_freq[new_idx] = cluster_freq[new_idx]+word_freq[word]
				cluster_entropy[old_idx] = -1*(-1*cluster_entropy[old_idx] - (word_freq[word]/float(cluster_freq[old_idx])))
				cluster_entropy[new_idx] = -1*(-1*cluster_entropy[new_idx] + (word_freq[word]/float(cluster_freq[new_idx])))
				cluster_freq[old_idx] = cluster_freq[old_idx]-word_freq[word]
				
				cluster_entropy_prob = cluster_entropy/sum(cluster_entropy)
		
		train_entropy,valid_entropy,test_entropy = get_separate_entorpy(train_cluster_freq,valid_cluster_freq,test_cluster_freq)	
		
		target_train.write(str(train_entropy)+"\n")
		target_valid.write(str(valid_entropy)+"\n")
		target_test.write(str(test_entropy)+"\n")
		
		target_train.flush()
		target_valid.flush()
		target_test.flush()
		iter += 1
	target_train.close()
	target_valid.close()
	target_test.close()
		
	uniform_clusters = defaultdict(list)
	for word in uniform_clusters_words.keys():
		idx = uniform_clusters_words[word]
		uniform_clusters[idx].append(word)
		
	print("Writing Rank-Clustering Files!")
	mcmc_file = codecs.open(base_path+dataset+"/mapping/"+lang+"-"+tgt+"/randommcmcmt_mapping.2gram."+lang,'w',encoding='utf-8')
	for key in uniform_clusters.keys():
		for word in uniform_clusters[key]:
			mcmc_file.write(word+","+str(key)+"\n")
	mcmc_file.close()
	
#####################################################################################################################

#metaphone_dict = copy.deepcopy(encoded_dict)
metaphone_dict = defaultdict(list)
for word in vocab.keys():
	encoded_word = jellyfish.metaphone(word)
	if encoded_word == '':
		metaphone_dict[word].append(word)
	else:
		metaphone_dict[encoded_word].append(word)
#rank_word_method(vocab_list,model,metaphone_dict,dist_matrix,vocab_dict)
#max_k_method(vocab_list,encoded_dict)
#rank_word_method_new(vocab_list,model,metaphone_dict,vocab_dict)
#print("DONE!")

#If I fix the number of clusters to be 20% of the vocab and then sample cluster size from Poisson distribution, then the number of words sampled is less than the total number of words.
#This is fixed if I do not restrict the number of clusters.

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
#for key in tqdm(metaphone_dict.keys()):
'''
biggest cluster is K
n/K points
for K size cluster	(forming one cluster)
	each from priliminary cluster
if j<K size cluster:
	sample a j/K %age cluster from primilinary cluster
K=10
j=5
sample 1/2 of clusters
(j/K)*(n/K)
'''
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
				cluster_list.append(new_cluster[0])
				del new_cluster[0]
				new_meta_dict[key2] = new_cluster				
		#key_i = sample_count*(float(n)/K)
		final_clusters[str(key_i)] = cluster_list
		key_i += 1
		print("Adding "+str(len(cluster_list))+" elements in cluster "+str(key_i))
	elif cluster_len==0:
		exit()
	else:
		j = cluster_len
		#sample_count = int(((j*1.0)/float(K))*((n)/float(K)))
		sample_count = int(((j/float(K)))*len(list(new_meta_dict.keys())))
		if sample_count == 0:
			exit()
		selected_keys = random.sample(new_meta_dict.keys(),sample_count)
		cluster_list = []
		for key2 in selected_keys:
			new_cluster = new_meta_dict[key]
			if len(new_cluster) > 0:
				cluster_list.append(new_cluster[0])
				del new_cluster[0]
				new_meta_dict[key2] = new_cluster
		#key_i = sample_count*(float(n)/K)
		#if len(cluster_list) == 0:
		#	exit()
		#else:
		final_clusters[str(key_i)] = cluster_list
		key_i += 1
		print("Adding "+str(len(cluster_list))+" elements in cluster "+str(key_i))
	current_count -= len(cluster_list)
	
'''
print("Writing Sample-Clustering Files!")
sample_file = codecs.open(base_path+"/data/randomsample_mapping."+lang,'w',encoding='utf-8')
sampe_data = defaultdict(list)
for key in final_clusters.keys():
	for word in final_clusters[key]:
		sample_file.write(word+","+str(key)+"\n")
		sample_data[key].append(word)
sample_file.close()
'''
print("DONE!")