import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, evaluate_model, process_samples,\
    ranking_sequence

model_path = 'model.pt'
batch_size = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = torch.load(model_path)
    training_data, testing_data, valid_data, all_relations, vocabulary,  embedding=\
        gen_data()
    acc=evaluate_model(model, testing_data, batch_size, all_relations, device)
    print('accuracy:', acc)
