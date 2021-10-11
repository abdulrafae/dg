import codecs
import sys

filename=sys.argv[1]
with codecs.open(filename,'r',encoding='utf-8') as in_data:
	for line in in_data:
		line = line.strip().split()
		for word in line:
			vocab[word] += 1
			
target = codecs.open('data/vocab.txt','w',encoding='utf-8')
for key in vocab.keys():
	target.write(key+" : "+str(vocab[key])+"\n")
target.close()
