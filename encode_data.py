import jellyfish
import argparse
import os
from shutil import copyfile as cp

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--source', '-s', type=str, help='Source language')
parser.add_argument('--target', '-t', type=str, help='Target language')
parser.add_argument('--mappingfile', '-m', type=str, help='Mapping file')
parser.add_argument('--inpath', '-i',  type=int, help='Input path')
parser.add_argument('--filenames', '-f', type=str, default='train,valid,test', help='Input filenames comma separated')
parser.add_argument('--output', '-o', type=str, help='Output path')
parser.add_argument('--algo', '-a', type=str, default='algo1', help='Algorithm format for file extension')
args = parser.parse_args()

src = args.source
tgt = args.target
inpath = args.infile
filenames = args.filenames.split(',')
mappingfile = args.mappingfile
inpath = args.inpath
outpath = args.outpath
algo = args.algo

groups = dict()
with codecs.open(mappingfile,'r',encoding='utf-8') as f:
	for line in f:
		parts = line.strip().split(' : ')
		word = parts[0]
		groupid = parts[1]
		groups[word] = groupid

for filename in filenames:
	print(src+" : "+format+" : "+filename)
	grouping_file = open(outpath+'/'+filename+'.'+algo,'w',encoding='utf-8')
	concat_file = open(outpath+'/'+filename+'.'+src+algo,'w',encoding='utf-8')
	with open(inpath+'/'+filename+'.'+src,'r') as input:
		for i,line in enumerate(input):
			line = line.strip()
			words = line.split()
			str = ""
			for word in words:
				groupid = ""
				
				try:
					groupid = groups[word]
				except:
					print("Group not found for "+word+".")
					exit()
				str += groupid + " "
			str = str[:-1]
			try:
				grouping_file.write(str+'\n')
				concat_file.write(line+' '+str+'\n')
			except:
				print("Line ",i)
				exit()
	grouping_file.close()
	concat_file.close()
	
	cp(inpath+filename+'.'+src,outpath+filename+'.'+src)
	if filename!='test':
		cp(inpath+filename+'.'+tgt,outpath+filename+'.'+tgt)
			
print("DONE!")
