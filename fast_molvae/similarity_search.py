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
prefix = '15'
model = JTNNVAE(vocab, 300, 56, 20, 3)
model.load_state_dict(torch.load('vae_model/model.iter-6500'))
model = model.cuda()


smile = 'O=C(O)c1ccc(cc1)c1c2ccc([NH]2)c(c2nc(C=C2)c(c2[NH]c(cc2)c(c2ccc(cc2)C(=O)O)c2C=Cc1n2)c1ccc(cc1)C(=O)O)c1ccc(cc1)C(=O)O'

val = model.encode_from_smiles([smile,])
x_tree_vecs = val[0][:300]
x_mol_vecs = val[0][300:]
z_tree_vecs,tree_kl = model.rsample(x_tree_vecs, model.T_mean, model.T_var)
z_mol_vecs,mol_kl = model.rsample(x_mol_vecs, model.G_mean, model.G_var)
z1 = z_tree_vecs.cpu().detach().numpy()
z2 = z_mol_vecs.cpu().detach().numpy()

print(list(z1)+list(z2))