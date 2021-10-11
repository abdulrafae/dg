from gensim.models import Word2Vec
import codecs
import sys

filename = sys.argv[1]
with codecs.open(filename,'r',encoding='utf-8') as in_data:
	for line in tqdm(in_data.readlines()):
		words = line.strip().split(' : ')
		data.append(words)

emb=100
model = Word2Vec(data, size=emb, window=5, min_count=1, workers=4)
model.save("data/w2v_models/"+lang+"_"+str(emb)+".model")
print("DONE!")
