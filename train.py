import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import quadprog
import random

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence,\
    copy_grad_data, get_grad_params
from evaluate import evaluate_model
from config import CONFIG as conf
#from continue_train import sample_constrains

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
model_path = conf['model_path']
device = conf['device']
lr = conf['learning_rate']
loss_margin = conf['loss_margin']
num_contrain = conf['num_constrain']
data_per_constrain = conf['data_per_constrain']
random.seed(100)

def sample_given_pro(sample_pro_set, num_samples):
    samples = list(sample_pro_set.keys())
    fre = np.array(list(sample_pro_set.values()))
    pro = fre/float(sum(fre))
    #print(samples, pro)
    #selected_sample = np.random.choice(samples, num_samples, True, pro)
    selected_sample = np.random.choice(samples, min(num_samples, len(samples)),
                                       False, pro)
    #print(selected_sample)
    return selected_sample

def sample_constrains(rel_samples, relations_frequences):
    selected_rels = sample_given_pro(relations_frequences, num_contrain)
    ret_samples = []
    for i in range(num_contrain):
        rel_index = selected_rels[i]
        ret_samples.append(random.sample(rel_samples[rel_index],
                                         min(data_per_constrain,
                                             len(rel_samples[rel_index]))))
    return ret_samples

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
    return all_scores, loss

# copied from facebook open scource. (https://github.com/facebookresearch/
# GradientEpisodicMemory/blob/master/model/gem.py)
def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().double().numpy()
    memories_np = memories_np[~np.all(memories_np == 0, axis=1)]
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    #print(memories_np.shape)
    #print(gradient.shape)
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    #print(memories_np)
    #print(P, q, G, h)
    v = quadprog.solve_qp(P, q, G, h)[0]
    #print(v)
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1))

def rescale_grad(grad, past_fisher):
    grad_np = grad.cpu().contiguous().view(-1).double().numpy()
    fisher_1d = torch.cat([item.view(-1) for item in past_fisher])
    fisher_np_1d = fisher_1d.cpu().double().numpy()
    reverse_fisher_1d = 1/fisher_np_1d
    #print(reverse_fisher_1d[:100])
    scale_fisher = reverse_fisher_1d / reverse_fisher_1d.max()
    #scale_fisher = (reverse_fisher_1d-reverse_fisher_1d.min())/\
    #    (reverse_fisher_1d.max()-reverse_fisher_1d.min())
    #print(scale_fisher.size, grad_np.size)
    return grad.copy_(torch.Tensor(np.multiply(grad_np, scale_fisher)).view(-1))

# copied from facebook open scource. (https://github.com/facebookresearch/
# GradientEpisodicMemory/blob/master/model/gem.py)
def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def get_grads_memory_data(model, memory_data, loss_function,
                          all_relations, device):
    memory_data_grads = []
    for data in memory_data:
        scores, loss = feed_samples(model, data,
                                    loss_function, all_relations, device)
        memory_data_grads.append(copy_grad_data(model))
        del scores
        del loss
        #print(memory_data_grads[-1][:10])
    if len(memory_data_grads) > 1:
        return torch.stack(memory_data_grads)
    elif len(memory_data_grads) == 1:
        return memory_data_grads[0].view(1,-1)
    else:
        return []

def check_constrain(memory_grads, sample_grad):
    sample_grad_ = torch.t(sample_grad.view(1, -1))
    result = torch.matmul(memory_grads, sample_grad_)
    if (result < 0).sum() != 0:
        return False
    else:
        return True

def train(training_data, valid_data, vocabulary, embedding_dim, hidden_dim,
          device, batch_size, lr, model_path, embedding, all_relations,
          model=None, epoch=100, memory_data=[], loss_margin=2.0,
          past_fisher=None, rel_samples=[], relation_frequences=[]):
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
            if len(rel_samples) > 0:
                memory_data = sample_constrains(rel_samples,
                                                relation_frequences)
            memory_data_grads = get_grads_memory_data(model, memory_data,
                                                      loss_function,
                                                      all_relations,
                                                      device)
            #print(memory_data_grads)
            samples = training_data[i*batch_size:(i+1)*batch_size]
            scores, loss = feed_samples(model, samples, loss_function,
                                        all_relations, device)
            sample_grad = copy_grad_data(model)
            if len(memory_data_grads) > 0:
                if not check_constrain(memory_data_grads, sample_grad):
                    project2cone2(sample_grad, memory_data_grads)
                    if past_fisher is None:
                        grad_params = get_grad_params(model)
                        grad_dims = [param.data.numel() for param in grad_params]
                        overwrite_grad(grad_params, sample_grad, grad_dims)
            if past_fisher is not None:
                sample_grad = rescale_grad(sample_grad, past_fisher)
                grad_params = get_grad_params(model)
                grad_dims = [param.data.numel() for param in grad_params]
                overwrite_grad(grad_params, sample_grad, grad_dims)
            optimizer.step()
            del scores
            del loss
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
