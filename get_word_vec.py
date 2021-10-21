from gensim.models import Word2Vec
import codecs
import sys
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--infile', '-i', type=str, help='Input file')
parser.add_argument('--outpath', '-o', type=str, help='Output path')
parser.add_argument('--emb', '-e', default=100, type=int, help='Embedding dimension')
args = parser.parse_args()

infile = args.infile
outpath = args.outpath
emb = args.emb
with codecs.open(infile,'r',encoding='utf-8') as in_data:
	for line in tqdm(in_data.readlines()):
		words = line.strip().split(' : ')
		data.append(words)

model = Word2Vec(data, size=emb, window=5, min_count=1, workers=4)
model.save(outpath+"/w2v_"+str(emb)+".model")
