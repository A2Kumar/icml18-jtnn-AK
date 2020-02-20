import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import pickle as pickle

from fast_jtnn import *
import rdkit
from tqdm import tqdm

vocab_path = './vocab.txt'

vocab = [x.strip("\r\n ") for x in open(vocab_path)] 
vocab = Vocab(vocab)

model = JTNNVAE(vocab, 300, 56, 20, 3)
model.load_state_dict(torch.load())
model = model.cuda()

with open('./keys.txt') as f:
    data = [line.strip("\r\n ").split()[0] for line in f]

print(len(data))
ans = model.encode_from_smiles(data)
