import codecs
import sys
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--infile', '-i', type=str, help='Input file')
parser.add_argument('--outpath', '-o', type=str, help='Output path')
args = parser.parse_args()

infile = args.infile
outpath = args.outpath

with codecs.open(infile,'r',encoding='utf-8') as in_data:
	for line in in_data:
		line = line.strip().split()
		for word in line:
			vocab[word] += 1
			
target = codecs.open(outpath+'/vocab.txt','w',encoding='utf-8')
for key in vocab.keys():
	target.write(key+" : "+str(vocab[key])+"\n")
target.close()
