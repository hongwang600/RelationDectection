import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import quadprog
import random
import time

from data import gen_data, read_origin_relation
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence,\
    copy_grad_data, get_grad_params
from evaluate import evaluate_model
from config import CONFIG as conf
from sklearn.cluster import KMeans
from sklearn import preprocessing  # to normalise existing X
#from continue_train import sample_constrains

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
model_path = conf['model_path']
device = conf['device']
lr = conf['learning_rate']
loss_margin = conf['loss_margin']
num_constrain = conf['num_constrain']
data_per_constrain = conf['data_per_constrain']
num_cands = conf['num_cands']
random.seed(100)
origin_relation_names = read_origin_relation()

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

def np_normalize(a):
    return (a-np.min(a))/np.ptp(a)

def sample_given_simi(similarity, num_sample):
    similarity = similarity.cpu().double().numpy()
    simi_pro = similarity/sum(similarity)
    indexs = list(range(len(simi_pro)))
    return np.random.choice(indexs, min(len(simi_pro), num_sample), False,
                            simi_pro)

def select_n_centers(sample_bert_embeds, start_seed_rel, num_centers =
                     num_constrain):
    sel_index = []
    #seed_rel_embeds = start_seed_rel
    seed_rel_embeds = \
        torch.from_numpy(preprocessing.normalize(start_seed_rel)).to(device)
    sample_bert_embeds = \
        torch.from_numpy(preprocessing.normalize(sample_bert_embeds)).to(device)
    cons_to_sel = min(len(sample_bert_embeds), num_centers)
    for i in range(cons_to_sel):
        #dis = nn.CosineSimilarity(dim=1)
        dis = torch.nn.PairwiseDistance()
        sample_similarity = []
        for seed_rel in seed_rel_embeds:
            sample_similarity.append(dis(sample_bert_embeds, seed_rel.view(1,-1)))
        sample_similarity = torch.stack(sample_similarity)
        #print(sample_similarity.size())
        #print(sample_similarity)
        sample_similarity, _ = sample_similarity.min(0)
        #this_index = sample_similarity.argmax()
        this_index = sample_given_simi(sample_similarity, 1)[0]
        sel_index.append(this_index)
        seed_rel_embeds = torch.cat((seed_rel_embeds,
                                     sample_bert_embeds[this_index].view(1,-1)))
    ret_sel = np.zeros(len(sample_bert_embeds))
    for i in sel_index:
        ret_sel[i] = 1
    #print(ret_sel)
    return ret_sel, sel_index

def mix_random_center(pro, num_seed, sample_bert_embeds):
    indexs = list(range(len(pro)))
    sel_seed_indexs = np.random.choice(indexs, min(len(indexs), num_seed),
                                       False, pro)
    seed_embeds = sample_bert_embeds[sel_seed_indexs]
    ret_sel, sel_index = select_n_centers(sample_bert_embeds, seed_embeds,
                                          num_constrain-num_seed)
    #print(sel_seed_indexs, sel_index, len(pro))
    return list(sel_seed_indexs) + sel_index

def select_data_kmeans(embeds, samples, num_sel_data):
    #que_embeds = get_que_embed(model, samples, all_relations)
    #print(que_embeds[:5])
    embeds_Norm = preprocessing.normalize(embeds)
    num_clusters = min(num_sel_data, len(samples))
    distances = KMeans(n_clusters=num_clusters,
                    random_state=0).fit_transform(embeds_Norm)
    selected_samples = []
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        selected_samples.append(samples[sel_index])
    return selected_samples

def kmeans_pro(embeds, samples, num_sel_data):
    #que_embeds = get_que_embed(model, samples, all_relations)
    #print(que_embeds[:5])
    embeds_Norm = preprocessing.normalize(embeds)
    num_clusters = min(num_sel_data, len(samples))
    distances = KMeans(n_clusters=num_clusters,
                    random_state=0).fit_transform(embeds_Norm)
    rel_dis = 1/np.exp(np.min(distances, 1))
    return rel_dis/sum(rel_dis)

def sample_given_pro_bert(sample_pro_set, num_samples, bert_rel_feature,
                          seed_rels, rel_acc_diff, given_pro):
    samples = list(sample_pro_set.keys())
    if given_pro is not None:
        return np.random.choice(samples, min(num_samples, len(samples)),
                                       False, given_pro)
    return random.sample(samples, min(len(samples), num_samples))
    sample_bert_embeds = torch.from_numpy(np.asarray(
        [bert_rel_feature[i] for i in samples])).to(device)
    seed_rel_embeds = torch.from_numpy(np.asarray(
        [bert_rel_feature[i] for i in seed_rels])).to(device)
    sample_embeds_np = sample_bert_embeds.cpu().double().numpy()
    #return select_data_kmeans(sample_embeds_np, samples, num_constrain)
    #'''
    seed_center = random.sample(list(range(len(samples))), 1)[0]
    seed_center_embed = sample_bert_embeds[seed_center]
    sel_rel, sel_index = select_n_centers(sample_bert_embeds,
                                          seed_center_embed.view(1,-1))
    sel_index += [seed_center]
    #sel_rel, sel_index = select_n_centers(sample_bert_embeds, seed_rel_embeds)
    #print(sel_index)
    return [samples[i] for i in sel_index]
    #'''
    #print(sample_bert_embeds.size(), seed_rel_embeds.size())
    '''
    dis = nn.CosineSimilarity(dim=1)
    #dis = torch.nn.PairwiseDistance()
    sample_similarity = []
    for seed_rel in seed_rel_embeds:
        sample_similarity.append(dis(sample_bert_embeds, seed_rel.view(1,-1)))
    sample_similarity = torch.stack(sample_similarity)
    #print(sample_similarity.size())
    #print(sample_similarity)
    #sample_similarity = sample_similarity.mean(0).cpu().double().numpy()
    sample_similarity, _ = sample_similarity.max(0)
    sample_similarity = sample_similarity.cpu().double().numpy()
    #sel_index = sample_similarity.argsort()[
    #    -(min(num_samples, len(samples))):]
    #return [samples[i] for i in sel_index]
    #sample_similarity += 1
    #sample_similarity = np.exp(sample_similarity)
    #sample_similarity = 1 - sample_similarity
    for i in range(len(sample_similarity)):
        sample_similarity[i] = max(0.001, sample_similarity[i])
    '''
    #print(sample_similarity)
    #sample_similarity = 1/sample_similarity
    #print(sample_similarity)
    #fre = np.array(list(sample_pro_set.values())) + sample_similarity*10
    #fre = np.array(list(sample_pro_set.values())) +\
    #    np_normalize(sample_similarity)*50
    #fre = np.array(list(sample_pro_set.values())) + sel_rel*20
    fre = np.array(list(sample_pro_set.values()))
    #fre = np.multiply(np.array(list(sample_pro_set.values())),
    #                  np.array(list(rel_acc_diff.values())))
    pro = fre/float(sum(fre))
    #sample_similarity = np.exp(sample_similarity)
    #pro = sample_similarity/sum(sample_similarity)
    #print('fre', fre/float(sum(fre)))
    #print('simi', pro)
    #sel_index = mix_random_center(pro, num_constrain//2, sample_bert_embeds)
    #print(sel_index)
    #return [samples[i] for i in sel_index]
    #print(samples, pro)
    #selected_sample = np.random.choice(samples, num_samples, True, pro)
    selected_sample = np.random.choice(samples, min(num_samples, len(samples)),
                                       False, pro)
    #print(selected_sample)
    return selected_sample

def get_nearest_cand(pos_rel, seen_rels, rel_embeds, cand_size, rm_cur = True):
    #return random.sample(samples, min(len(samples), num_samples))
    if rm_cur:
        samples = [rel for rel in seen_rels if rel!=pos_rel]
    else:
        samples = seen_rels
    sample_bert_embeds = torch.from_numpy(np.asarray(
        [rel_embeds[i] for i in samples])).to(device)
    seed_rel_embeds = torch.from_numpy(np.asarray(
        [rel_embeds[pos_rel]])).to(device)
    #dis = torch.nn.PairwiseDistance()
    dis = nn.CosineSimilarity(dim=1)
    sample_similarity = dis(sample_bert_embeds, seed_rel_embeds.view(1,-1))
    sample_similarity = sample_similarity.cpu().double().numpy()
    sel_index = sample_similarity.argsort()[
        -(min(5*cand_size, len(samples))):]
    sel_index = random.sample(sel_index, min(cand_size, len(sel_index)))
    '''
    print('pos: ', origin_relation_names[pos_rel])
    for i in sel_index:
        print(origin_relation_names[samples[i]])
        '''
    #print(sample_similarity)
    #print(sel_index)
    return [samples[i] for i in sel_index]
    #print(sample_similarity.size())
    #print(sample_similarity)
    #sample_similarity = sample_similarity.mean(0).cpu().double().numpy()
    #sample_similarity, _ = sample_similarity.max(0)
    #sample_similarity = sample_similarity.cpu().double().numpy()

def update_rel_cands(memory_data, all_seen_cands, rel_embeds):
    if len(memory_data) >0:
        for this_memory in memory_data:
            for sample in this_memory:
                valid_rels = [rel for rel in all_seen_cands if rel!=sample[0]]
                sample[1] = random.sample(valid_rels,
                                        min(num_cands,len(valid_rels)))
                #print('random', sample[1])
                #sample[1] = get_nearest_cand(sample[0], valid_rels,
                #                                      rel_embeds, num_cands,
                #                                      False)
                #print('near', sample[1])

def sample_constrains(rel_samples, relations_frequences, rel_embeds,
                      seed_rels, rel_ques_cand, rel_acc_diff, given_pro,
                      all_seen_rels):
    selected_rels = sample_given_pro_bert(relations_frequences, num_constrain,
                                          rel_embeds, seed_rels,
                                          rel_acc_diff, given_pro)
    ret_samples = []
    for i in range(min(num_constrain, len(selected_rels))):
        rel_index = selected_rels[i]
        ret_samples.append(random.sample(rel_samples[rel_index],
                                         min(data_per_constrain,
                                             len(rel_samples[rel_index]))))
    #return ret_samples
    #all_cands = list(rel_ques_cand.keys())[:]
    for this_memory in ret_samples:
        for i, sample in enumerate(this_memory):
            '''
            cand_set = get_nearest_cand(sample[0], all_seen_rels, rel_embeds,
                                        num_cands)
                                        #num_cands)
                                        #len(all_seen_rels)//4)
            this_memory[i] = [sample[0],
                              random.sample(cand_set,
                                            min(len(cand_set),num_cands)),
                              sample[2]]
                              '''
            this_memory[i] = [sample[0], random.sample(
                all_seen_rels, min(len(all_seen_rels), num_cands)), sample[2]]
            if this_memory[i][0] in this_memory[i][1]:
                this_memory[i][1].remove(this_memory[i][0])
                #cand_set, min(len(cand_set), num_cands)), sample[2]]
    return ret_samples

def feed_samples(model, samples, loss_function, all_relations, device,
                 reverse_model=None, memory_que_embed=[], memory_rel_embed):
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
    #start_time = time.time()
    memories_np = memories.cpu().double().numpy()
    '''
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
    '''
    gradient_np = gradient.cpu().contiguous().view(1, -1).double().numpy()
    all_gradient = np.concatenate([memories_np, gradient_np])
    x = np.mean(all_gradient, 0)
    gradient.copy_(torch.Tensor(x).view(-1))
    #end_time = time.time()
    #print('proj time:', end_time-start_time)

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
          model=None, epoch=100, memory_data=[], loss_margin=0.5,
          past_fisher=None, rel_samples=[], relation_frequences=[],
          rel_embeds=None, rel_ques_cand=None, rel_acc_diff=None,
          all_seen_rels=None, update_rel_embed=None, reverse_model=None,
          memory_que_embed=[],memory_rel_embed=[]):
    if model is None:
        torch.manual_seed(100)
        model = SimilarityModel(embedding_dim, hidden_dim, len(vocabulary),
                                np.array(embedding), 1, device)
    loss_function = nn.MarginRankingLoss(loss_margin)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    #acc_pre=evaluate_model(model, valid_data, batch_size, all_relations, device)
    given_pro = None
    for epoch_i in range(epoch):
        #print('epoch', epoch_i)
        #training_data = training_data[0:100]
        for i in range((len(training_data)-1)//batch_size+1):
            samples = training_data[i*batch_size:(i+1)*batch_size]
            seed_rels = []
            for item in samples:
                if item[0] not in seed_rels:
                    seed_rels.append(item[0])
                '''
                for item_cand in item[1]:
                    if item_cand not in seed_rels:
                        seed_rels.append(item_cand)
                '''

            if len(rel_samples) > 0:
                memory_data = sample_constrains(rel_samples,
                                                relation_frequences,
                                                rel_embeds,
                                                seed_rels, rel_ques_cand,
                                                rel_acc_diff,
                                                given_pro, all_seen_rels)
            to_train_mem = memory_data
            if len(memory_data) > num_constrain:
                to_train_mem = random.sample(memory_data, num_constrain)
            memory_data_grads = get_grads_memory_data(model, to_train_mem,
                                                      loss_function,
                                                      all_relations,
                                                      device)
            #print(memory_data_grads)
            #start_time = time.time()
            scores, loss = feed_samples(model, samples, loss_function,
                                        all_relations, device, reverse_model,
                                        memory_que_embed, memory_rel_embed)
            #end_time = time.time()
            #print('forward time:', end_time - start_time)
            sample_grad = copy_grad_data(model)
            if len(memory_data_grads) > 0:
                #if not check_constrain(memory_data_grads, sample_grad):
                if True:
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
            if(epoch_i%5==0) and len(relation_frequences)>0 and False:
                update_rel_embed(model, all_seen_rels, all_relations, rel_embeds)
                samples = list(relation_frequences.keys())
                #return random.sample(samples, min(len(samples), num_samples))
                sample_embeds = torch.from_numpy(np.asarray(
                    [rel_embeds[i] for i in samples]))
                #seed_rel_embeds = torch.from_numpy(np.asarray(
                #    [rel_embeds[i] for i in seed_rels])).to(device)
                sample_embeds_np = sample_embeds.cpu().double().numpy()
                #given_pro = kmeans_pro(sample_embeds_np, samples, num_constrain)
            if epoch_i%5==0 and False:
                update_rel_embed(model, all_seen_rels, all_relations, rel_embeds)
                #update_rel_cands(memory_data, all_seen_rels, rel_embeds)
            del scores
            del loss
            '''
        acc=evaluate_model(model, valid_data, batch_size, all_relations, device)
        if acc > best_acc:
            torch.save(model, model_path)
    best_model = torch.load(model_path)
    return best_model
    '''
    #acc_aft=evaluate_model(model, valid_data, batch_size, all_relations, device)
    #return model, max(0, acc_aft-acc_pre)
    return model, 0

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
