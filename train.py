import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence
from evaluate import evaluate_model
from config import CONFIG as conf

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
model_path = conf['model_path']
device = conf['device']
lr = conf['learning_rate']
loss_margin = conf['loss_margin']

def feed_samples(model, samples, loss_function, all_relations, device):
    questions, relations, relation_set_lengths = process_samples(
        samples, all_relations, device)
    #print('got data')
    ranked_questions, reverse_question_indexs = \
        ranking_sequence(questions)
    ranked_relations, reverse_relation_indexs =\
        ranking_sequence(relations)
    question_lengths = [len(question) for question in ranked_questions]
    relation_lengths = [len(relation) for relation in ranked_relations]
    #print(ranked_questions)
    pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
    pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
    #print(pad_questions)
    pad_questions = pad_questions.to(device)
    pad_relations = pad_relations.to(device)
    #print(pad_questions)

    model.zero_grad()
    model.init_hidden(device, sum(relation_set_lengths))
    all_scores = model(pad_questions, pad_relations, device,
                       reverse_question_indexs, reverse_relation_indexs,
                       question_lengths, relation_lengths)
    all_scores = all_scores.to('cpu')
    pos_scores = []
    neg_scores = []
    start_index = 0
    for length in relation_set_lengths:
        pos_scores.append(all_scores[start_index].expand(length-1))
        neg_scores.append(all_scores[start_index+1:start_index+length])
        start_index += length
    pos_scores = torch.cat(pos_scores)
    neg_scores = torch.cat(neg_scores)

    loss = loss_function(pos_scores, neg_scores,
                         torch.ones(sum(relation_set_lengths)-
                                    len(relation_set_lengths)))
    loss.backward()


def train(training_data, valid_data, vocabulary, embedding_dim, hidden_dim,
          device, batch_size, lr, model_path, embedding, all_relations,
          model=None, epoch=100, loss_margin=2.0):
    if model is None:
        torch.manual_seed(100)
        model = SimilarityModel(embedding_dim, hidden_dim, len(vocabulary),
                                np.array(embedding), 1, device)
    loss_function = nn.MarginRankingLoss(loss_margin)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    for epoch_i in range(epoch):
        #print('epoch', epoch_i)
        #training_data = training_data[0:100]
        for i in range((len(training_data)-1)//batch_size+1):
            samples = training_data[i*batch_size:(i+1)*batch_size]
            feed_samples(model, samples, loss_function, all_relations, device)
            optimizer.step()
        '''
        acc=evaluate_model(model, valid_data, batch_size, all_relations, device)
        if acc > best_acc:
            torch.save(model, model_path)
    best_model = torch.load(model_path)
    return best_model
    '''
    return model

if __name__ == '__main__':
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    train(training_data, valid_data, vocabulary, embedding_dim, hidden_dim,
          device, batch_size, lr, model_path, embedding, all_relations,
          model=None, epoch=100)
    #print(training_data[0:10])
    #print(testing_data[0:10])
    #print(valid_data[0:10])
    #print(all_relations[0:10])
