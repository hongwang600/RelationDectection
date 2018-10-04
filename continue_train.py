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
from config import CONFIG as conf

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
device = conf['device']
num_clusters = conf['num_clusters']

def split_data(data_set, cluster_labels, num_clusters):
    pass

if __name__ == '__main__':
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    cluster_labels = cluster_data(num_clusters)
