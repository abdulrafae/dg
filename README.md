# Grouping Words with Semantic Diversity
Abdul Rafae Khan<sup>+</sup>, Karine Chubarian<sup>++</sup>, Anastasios Sidiropoulos<sup>++</sup> & Jia Xu<sup>+</sup>

<sup>+</sup> Stevens Institute of Technology

<sup>++</sup> University of Illinois at Chicago

## (1) Neural Machine Translation (NMT)

### Install dependencies
Setup fairseq toolkit
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

Install evaluation packages
```
pip install sacrebleu
```
Install tokenization packages
```
git clone https://github.com/moses-smt/mosesdecoder.git
```

Install Byte-pair Encoding pacakges
```
git clone https://github.com/rsennrich/subword-nmt.git
```

### Download and Pre-process IWSLT'17 French-English data

Download and tokenize data
```
bash prepare_data.sh
```

Train word vectors
```
python get_word_vec.py --infile data/train.en --outpath models/word2vec/ --emb 100
```

Create groups
```python algorithm1.py --infile data/train.en --outpath groupings/ 
python algorithm2.py --infile data/train.en --outpath groupings/ 
python algorithm3.py --infile data/train.en --outpath groupings/ --wordvecfile models/word2vec/w2v_100.model (use --matrixfile path/to/distance/matrix/file) 
python algorithm4.py --infile data/train.en --outpath groupings/ 
```

Encode files
```
python encode_data.py --source en --target fr --algo algo1 --groupingfile groupings/algo1_grouping.txt --inpath data/ --filenames train,valid,test --outpath data/algo1en_fr/ 
```

Byte-pair encode the data
```
bash apply_bpe.sh en algo fr (e.g. algo1,algo2,algo3 or algo4)
```

### Train NMT System
Train French+Algorithm1-English Concatenation Model
```
bash train_concatenation.sh en algo fr (e.g. algo1,algo2,algo3 or algo4)
```

## (2) Coding Language Modeling

### Install dependencies
```
git clone https://github.com/facebookresearch/XLM.git
```

### Download and Pre-process IWSLT'17 French-English data

Download and tokenize data
```
bash prepare_data.sh
```

Create vocabulary 
```
OUTPATH=data/processed/XLM_en/30k
mkdir -p $OUTPATH

python get_vocab.py --infile $OUTPATH/train.en --outfile $OUTPATH/vocab.en
python get_vocab.py --infile $OUTPATH/train.enalgo1 --outfile $OUTPATH/vocab.algo1
```

Binarize data
```
python XLM/preprocess.py $OUTPATH/vocab.en $OUTPATH/train.en &
python XLM/preprocess.py $OUTPATH/vocab.en $OUTPATH/valid.en &
python XLM/preprocess.py $OUTPATH/vocab.en $OUTPATH/test.en &

python XLM/preprocess.py $OUTPATH/vocab.algo1 $OUTPATH/train.algo1 &
python XLM/preprocess.py $OUTPATH/vocab.algo1 $OUTPATH/valid.algo1 &
python XLM/preprocess.py $OUTPATH/vocab.algo1 $OUTPATH/test.algo1 &
```

Train English baseline
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name xlm_en --dump_path ./dumped_xlm_en --data_path $OUTPATH --lgs 'en' --clm_steps '' --mlm_steps 'en' --emb_dim 256 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 32 --bptt 256 --optimizer adam_inverse_sqrt,lr=0.00010,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001 --epoch_size 300000 --max_epoch 100000 --validation_metrics _valid_en_mlm_ppl --stopping_criterion _valid_en_mlm_ppl,25 --fp16 true --word_mask_keep_rand '0.8,0.1,0.1' --word_pred '0.15' 
```

Train English+Algorithm1
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name xlm_en_ny --dump_path ./dumped_xlm_en_ny --data_path $OUTPATH --lgs 'en' --clm_steps '' --mlm_steps 'en,ny' --emb_dim 256 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 32 --bptt 256 --optimizer adam_inverse_sqrt,lr=0.00010,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001 --epoch_size 300000 --max_epoch 100000 --validation_metrics _valid_en_mlm_ppl --stopping_criterion _valid_en_mlm_ppl,25 --fp16 true --word_mask_keep_rand '0.8,0.1,0.1' --word_pred '0.15' 
```

## (2) Coding Language Modeling

### Install dependencies
```
pip install keras
```

### Run Baseline
```
python make_model.py --save-path baseline/ --alpha 0.0 
```

### Run Algorithm1 (Alpha=0.5)
```
python make_model.py --save-path algo1/ --alpha 0.5
```
