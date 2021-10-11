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

lang = 'en'
src = 'en'
tgt = 'fr'
emb = 2
	
data = []
base_path = 'IWSLT/data/en-fr_val/'

vocab_py_to_char = defaultdict(set)
vocab_char_to_py = defaultdict(set)

filenames = ['test','valid','train']
for filename in filenames:
	with codecs.open(base_path+src+"_"+tgt+"/"+filename+'.'+src,'r',encoding='utf-8') as in_data:
		for line in in_data:
			line = line.strip().split()	
			data.append(line)

model = Word2Vec(data, size=emb, window=5, min_count=1, workers=4)

vocab = model.wv.vocab.keys()	
	
for word in model.wv.vocab.keys():
	py = jellyfish.metaphone(word)
	if py == '':
		py = word
	vocab_py_to_char[py].add(word)
	vocab_char_to_py[word].add(py)
	
K = len(vocab_py_to_char.keys())
	
tmp = copy.deepcopy(list(vocab))	
print(K)
print(len(tmp))
#exit()
random_mapping = dict()
method = 'kmeans'
####################
#####FIT RANDMT#####
####################	
if method == 'randmt':
	coding='randmt'
	src_zh = src+coding
	lengths = []
	for key in vocab_py_to_char.keys():
		size = len(vocab_py_to_char[key])
		lengths.append(size)
	
	size_hist = Counter(lengths)

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

####################
#####FIT RANDUNI####
####################
elif method == 'randuni':
	coding='randuni'	
	src_zh = src+coding
	while len(tmp)!=0:
		id = random.randint(0,K)
		word = random.choice(tmp)
		random_mapping[word] = id
		tmp.remove(word)

####################
#####FIT KMeans#####
####################
elif method == 'kmeans':
	coding='kmeans'
	src_zh = src+coding
	points = np.zeros((len(model.wv.vocab.keys()),emb))
	for j,word in enumerate(tmp):
		points[j,:] = model.wv[word]
	batch = K
	kmeans = MiniBatchKMeans(n_clusters=K,random_state=0,batch_size=batch)
	for i in tqdm(range(0,points.shape[0],batch)):
		kmeans = kmeans.partial_fit(points[i:i+batch,:])
	'''
	kmeans = KMeans(n_clusters=K)
	kmeans.fit(points)
	'''
	labels = kmeans.predict(points)

	for i,word in enumerate(vocab):
		random_mapping[word] = labels[i]

else:
	print("Method Incorrect!")
	exit()

target = codecs.open('IWSLT/mapping/en-fr/'+coding+'_mapping.en','w',encoding='utf-8')
for key in random_mapping.keys():
	target.write(str(key)+" : "+str(random_mapping[key])+"\n")
target.close()

filenames = ['test','valid','train']
src = 'en'
tgt = 'fr'
#src_zh = src+'kmeans'

for filename in filenames:
	#target_zh = codecs.open('New/zh-en_processed/zh_en/'+filename+'.zh','w',encoding='utf-8')
	target_zhpy = codecs.open(base_path+src_zh+'_'+tgt+'/'+filename+'.'+src_zh,'w',encoding='utf-8')
	target_py = codecs.open(base_path+coding+'_'+tgt+'/'+filename+'.'+coding,'w',encoding='utf-8')
	print(base_path+src_zh+'_'+tgt+'/'+filename+'.'+src_zh)
	print(base_path+coding+'_'+tgt+'/'+filename+'.'+coding)
	with codecs.open(base_path+src+"_"+tgt+"/"+filename+'.'+src,'r',encoding='utf-8') as data:
		for i,line in enumerate(data):
			print(i)
			line = line.strip()
			words = line.split()
			str_py = ""
			
			for word in words:
				str_py += str(random_mapping[word]) + " "

			str_py = str_py[:-1]
			target_zhpy.write(line+" "+str_py+'\n')
			target_py.write(str_py+'\n')
	target_zhpy.close()
	target_py.close()
