## Environment

- tqdm==4.59.0
- numpy==1.20.1
- scikit-learn==0.24.1
- scipy==1.6.2
- torch==1.9.0


## Download Datasets

You can download the datasets from https://drive.google.com/file/d/1uzyA1liRRqrrrq3oeL-ZIVOr1ot1k0et/view?usp=sharing. 

## Before Experiments

Install the kbc package.
```
python setup.py install
```

Preprocess the datasets.
```
python twq/process_icews.py
python twq/process_gdelt.py 
```

## Run the Experiments

```
python twq/learner.py --dataset ICEWS14 --model TWQ_main --rank 1200 --valid_freq 1 --max_epoch 20 --learning_rate 0.05 --batch_size 1000  --cycle 30 --gpu 0
python twq/learner.py --dataset ICEWS05-15 --model TWQ_main --rank 1200 --valid_freq 1 --max_epoch 20 --learning_rate 0.1 --batch_size 1000  --cycle 120 --gpu 0
python twq/learner.py --dataset gdelt --model TWQ_main --rank 1500 --valid_freq 10 --max_epoch 20 --learning_rate 0.1 --batch_size 1000  --cycle 365 --gpu 0
```
## Multi-hop Queries and Case Study
When performing multi-hop queries and case study running, you should also manually modify the parameters of the following files: datasets.py, learner.py, models.py, and process_icews.py according to the instructions in these files.

```
python twq/learner.py --dataset ICEWS14-1P --model TWQ_main --rank 1200 --valid_freq 1 --max_epoch 20 --learning_rate 0.05 --batch_size 1000  --cycle 30 --gpu 0
python twq/learner.py --dataset ICEWS14-2P --model TWQ_main --rank 1200 --valid_freq 1 --max_epoch 20 --learning_rate 0.05 --batch_size 1000  --cycle 30 --gpu 0
python twq/learner.py --dataset ICEWS14-3P --model TWQ_main --rank 1200 --valid_freq 1 --max_epoch 20 --learning_rate 0.05 --batch_size 1000  --cycle 30 --gpu 0
```

