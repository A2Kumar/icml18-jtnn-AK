import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import pandas as pd
import argparse
from collections import deque
import pickle as pickle

from fast_jtnn import *
import rdkit
from tqdm import tqdm

vocab_path = './vocab.txt'

vocab = [x.strip("\r\n ") for x in open(vocab_path)] 
vocab = Vocab(vocab)
prefix = '07'
model = JTNNVAE(vocab, 500, 128, 20, 3)
model.load_state_dict(torch.load('vae_model/model.iter-9500'))
model = model.cuda()

with open('./keys'+prefix+'.txt') as f:
    data = [line.strip("\r\n ").split()[0] for line in f]

print(len(data))
ans = []
error_num = []
for num,k in tqdm(enumerate(data)):
	try:
		val = model.encode_from_smiles([k,])
		ans.append((k,val))
	except Exception as e:
		print('Error on:',num,e)
		error_num.append(num)

results = {}
for num,k in enumerate(ans):
	if num in error_num:
		print('Skipping:',num)
		continue
	x_tree_vecs = k[1][0][:500]
	x_mol_vecs = k[1][0][500:]
	z_tree_vecs,tree_kl = model.rsample(x_tree_vecs, model.T_mean, model.T_var)
	z_mol_vecs,mol_kl = model.rsample(x_mol_vecs, model.G_mean, model.G_var)
	z1 = z_tree_vecs.cpu().detach().numpy()
	z2 = z_mol_vecs.cpu().detach().numpy()
	results[k[0]] = (z1,z2)


vae_features = pd.DataFrame.from_dict(results,orient='index')
vae_features.to_csv('./vae_features'+prefix+'.csv')