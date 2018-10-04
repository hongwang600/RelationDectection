'''
relation_file = "./data/relation.2M.list"
training_file = "./data/train.replace_ne.withpool"
test_file = "./data/test.replace_ne.withpool"
valid_file = "./data/valid.replace_ne.withpool"
#glove_file = "./data/glove.6B.300d.txt"
glove_file = "./data/small_glove.6B.300d.txt"
embedding_size = 300
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, evaluate_model, process_samples,\
    ranking_sequence
from data_partition import cluster_data
embedding_dim = 300
hidden_dim = 200
batch_size = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_clusters = 20

if __name__ == '__main__':
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    cluster_labels = cluster_data(num_clusters)
