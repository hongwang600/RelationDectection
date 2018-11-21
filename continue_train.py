import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import time
from sklearn.cluster import KMeans

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence,\
    get_grad_params, copy_param_data, copy_grad_data
from evaluate import evaluate_model, compute_diff_scores
from data_partition import cluster_data
from config import CONFIG as conf
from train import train, sample_constrains, sample_given_pro, get_nearest_cand,\
    select_data_kmeans
from compute_rel_embed import compute_rel_embed

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
device = conf['device']
num_clusters = conf['num_clusters']
lr = conf['learning_rate']
model_path = conf['model_path']
epoch = conf['epoch']
random_seed = conf['random_seed']
task_memory_size = conf['task_memory_size']
loss_margin = conf['loss_margin']
sequence_times = conf['sequence_times']
num_cands = conf['num_cands']
num_steps = conf['num_steps']
num_contrain = conf['num_constrain']
data_per_constrain = conf['data_per_constrain']

def split_data(data_set, cluster_labels, num_clusters, shuffle_index):
    splited_data = [[] for i in range(num_clusters)]
    for data in data_set:
        cluster_number = cluster_labels[data[0]]
        index_number = shuffle_index[cluster_number]
        splited_data[index_number].append(data)
    return splited_data

# remove unseen relations from the dataset
def remove_unseen_relation(dataset, seen_relations):
    cleaned_data = []
    for data in dataset:
        neg_cands = [cand for cand in data[1] if cand in seen_relations]
        if len(neg_cands) > 0:
            #data[1] = neg_cands
            #cleaned_data.append(data)
            cleaned_data.append([data[0], neg_cands, data[2]])
    return cleaned_data

def rm_unseen_rels(full_rel_samples, seen_relations):
    ret_rel_samples = {}
    for rel in full_rel_samples:
        ret_rel_samples[rel] = remove_unseen_relation(full_rel_samples[rel],
                                                       seen_relations)
    return ret_rel_samples

def print_list(result):
    for num in result:
        sys.stdout.write('%.3f, ' %num)
    print('')

def add_co_occur_edges(rel_ques_cand, all_rels):
    for this_rel in all_rels:
        if this_rel not in rel_ques_cand:
            rel_ques_cand[this_rel] = [{}, []]
        for cand_rel in all_rels:
            if cand_rel != this_rel:
                if cand_rel not in rel_ques_cand[this_rel][0]:
                    rel_ques_cand[this_rel][0][cand_rel] = 1
                else:
                    rel_ques_cand[this_rel][0][cand_rel] += 1

def enlarge_rel_graph(train_data, relations_frequences, rel_ques_cand):
    #new_relation_frequences = {}
    for sample in train_data:
        pos_index = sample[0]
        neg_cands = sample[1][:]
        question = sample[2]
        if relations_frequences is not None:
            if pos_index not in relations_frequences:
                relations_frequences[pos_index] = 1
            else:
                relations_frequences[pos_index] += 1
                relations_frequences[pos_index] = \
                    max(50, relations_frequences[pos_index])
        all_rels = [pos_index] + neg_cands
        add_co_occur_edges(rel_ques_cand, all_rels)
        if len(rel_ques_cand[pos_index][1]) < 10:
            rel_ques_cand[pos_index][1].append(question)

def updata_saved_relations(current_train_data, rel_samples,
                           relations_frequences, rel_acc_diff, acc_diff):
    for sample in current_train_data:
        pos_index = sample[0]
        if pos_index not in relations_frequences:
            relations_frequences[pos_index] = 1
            rel_acc_diff[pos_index] = acc_diff + 0.0001
            rel_samples[pos_index] = [sample]
        else:
        #elif len(rel_samples[pos_index]) < 10:
            relations_frequences[pos_index] = \
                max(50, relations_frequences[pos_index]+1)
            rel_samples[pos_index].append(sample)

def updata_full_saved_relations(current_train_data, rel_samples):
    for sample in current_train_data:
        pos_index = sample[0]
        if pos_index not in rel_samples:
            rel_samples[pos_index] = [sample]
        else:
            rel_samples[pos_index].append(sample)

def walk_n_steps(rel_ques_cand, num_steps, rel):
    for i in range(num_steps):
        rel = sample_given_pro(rel_ques_cand[rel][0], 1)[0]
    #print(rel)
    return rel

def random_walk(rel_ques_cand, num_cands, rel):
    '''
    cand_set = []
    #cur_rel = rel
    cur_num_cands = 0
    #print(rel)
    #print(rel_ques_cand[rel])
    while cur_num_cands < num_cands:
        end_rel = walk_n_steps(rel_ques_cand, num_steps, rel)
        #print(end_rel)
        if end_rel != rel:
            cand_set.append(end_rel)
        cur_num_cands += 1
    #print(cand_set)
    return cand_set
    '''
    all_cands = list(rel_ques_cand.keys())[:]
    return random.sample(all_cands, min(len(all_cands), num_cands))
    '''
    all_cands = list(rel_ques_cand.keys())[:]
    neighbor_cand = list(rel_ques_cand[rel][0].keys())
    all_cands = [cand for cand in all_cands if cand not in neighbor_cand]
    return random.sample(all_cands, min(len(all_cands), num_cands))
    '''

def sample_near_data(rel_ques_cand, current_train_data, num_samples):
    current_num_samples = 0
    current_index = 0
    num_training_samples = len(current_train_data)
    selected_rels = []
    remain_samples = num_samples
    samples = []
    while remain_samples > 0:
        this_sample = min(len(current_train_data), remain_samples)
        samples += random.sample(current_train_data, this_sample)
        remain_samples -= this_sample
    for this_sample in samples:
        end_rel = walk_n_steps(rel_ques_cand, num_steps, this_sample[0])
        selected_rels.append(end_rel)
        current_index = (current_index+1)%num_training_samples
    return selected_rels

def sample_away_data(rel_ques_cand, current_train_data, num_samples,
                     relations_frequences):
    current_num_samples = 0
    current_index = 0
    num_training_samples = len(current_train_data)
    selected_rels = []
    remain_samples = num_samples
    samples = []
    relation_list = list(relations_frequences.keys())
    while remain_samples > 0:
        this_sample = min(len(current_train_data), remain_samples)
        samples += random.sample(current_train_data, this_sample)
        remain_samples -= this_sample
    for this_sample in samples:
        #end_rel = walk_n_steps(rel_ques_cand, num_steps, this_sample[0])
        end_rel = this_sample[0]
        while end_rel == this_sample[0]:
            end_rel = random.sample(relation_list, 1)[0]
        selected_rels.append(end_rel)
        current_index = (current_index+1)%num_training_samples
    return selected_rels

def sample_relations(relations_frequences, rel_ques_cand, num_samples,
                     current_train_data):
    selected_rels = sample_given_pro(relations_frequences, num_samples)
    #selected_rels = sample_near_data(rel_ques_cand, current_train_data,
    #                                 num_samples)
    #selected_rels = sample_away_data(rel_ques_cand, current_train_data,
    #                                num_samples, relations_frequences)
    ret_relations = []
    for rel in selected_rels:
        #print(rel_ques_cand[rel])
        if len(rel_ques_cand[rel][1]) > 0:
            question = random.sample(rel_ques_cand[rel][1],1)[0]
            neg_cands = random_walk(rel_ques_cand, num_cands, rel)
            if len(neg_cands) > 0:
                ret_relations.append([rel, neg_cands, question])
    #print(ret_relations)
    return ret_relations

def sample_relations_task(relations_frequences, rel_ques_cand, num_samples):
    ret_samples = []
    for relation_fre in relations_frequences:
        task_sample = sample_relations(relation_fre, rel_ques_cand, num_samples)
        ret_samples.append(task_sample)
    return ret_samples

def gen_fisher(model, train_data, all_relations):
    num_correct = 0
    #testing_data = testing_data[0:100]
    softmax_func = nn.LogSoftmax(0)
    loss_func = nn.NLLLoss()
    fisher_batch_size = 1
    batch_epoch = (len(train_data)-1)//fisher_batch_size+1
    fisher = None
    for i in range(batch_epoch):
        model.zero_grad()
        losses = []
        samples = train_data[i*fisher_batch_size:(i+1)*fisher_batch_size]
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

        model.init_hidden(device, sum(relation_set_lengths))
        all_scores = model(pad_questions, pad_relations, device,
                           reverse_question_indexs, reverse_relation_indexs,
                           question_lengths, relation_lengths)
        all_scores = all_scores.to('cpu')
        start_index = 0
        for length in relation_set_lengths:
            scores = all_scores[start_index:start_index+length]
            start_index += length
            #print(scores)
            losses.append(loss_func(softmax_func(scores).view(1, -1),
                                    torch.tensor([0])))
        loss_batch = sum(losses)
        #print(loss_batch)
        loss_batch.backward()
        grad_params = get_grad_params(model)
        #for param in grad_params:
         #   print(param.grad)
        if fisher is None:
            fisher = [param.grad**2/batch_epoch
                         for param in grad_params]
        else:
            fisher = [fisher[i]+param.grad**2/batch_epoch
                         for i,param in enumerate(grad_params)]

    return fisher

def get_mean_fisher(model, train_data, all_relations):
    grad_params = get_grad_params(model)
    grad_mean = copy_param_data(grad_params)
    grad_fisher = gen_fisher(model, train_data, all_relations)
    return grad_mean, grad_fisher

def update_fisher(model, train_data, all_relations,
                  past_fisher, num_past_data):
    cur_mean, cur_fisher = get_mean_fisher(model, train_data, all_relations)
    num_cur_data = len(train_data)
    num_total_data = num_past_data + num_cur_data
    if past_fisher is None:
        past_fisher = cur_fisher
    elif cur_fisher is not None:
        for i in range(len(past_fisher)):
            #past_fisher[i] = past_fisher[i]*num_past_data/num_total_data +\
            #    cur_fisher[i]*num_cur_data/num_total_data
            past_fisher[i] = torch.max(past_fisher[i], cur_fisher[i])
    return past_fisher, num_total_data

def filter_data(data, model, all_relations):
    diff_scores = compute_diff_scores(model, data, batch_size, all_relations,
                                      device)
    selected_index = np.argsort(diff_scores)[0:len(data)*9//10]
    return [data[i] for i in selected_index]

def update_rel_embed(model, all_seen_rels, all_relations, rel_embeds):
    if model is not None and len(all_seen_rels) > 0:
        for i in range((len(all_seen_rels)-1)//batch_size+1):
            seen_rels_batch = all_seen_rels[i*batch_size:(i+1)*batch_size]
            relations = [torch.tensor(all_relations[i],
                                          dtype=torch.long).to(device)
                             for i in seen_rels_batch]
            model.init_hidden(device, len(relations))
            ranked_relations, reverse_relation_indexs = \
                ranking_sequence(relations)
            relation_lengths = [len(relation) for relation in ranked_relations]
            #print(ranked_relations)
            pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
            new_rel_embeds = model.compute_que_embed(pad_relations, relation_lengths,
                                                 reverse_relation_indexs)
            for i, rel in enumerate(seen_rels_batch):
                rel_embeds[rel] = new_rel_embeds[i].cpu().numpy()

def save_rel_embeds(model, all_seen_rels, all_relations, file_name):
    rel_embeds = {}
    if model is not None and len(all_seen_rels) > 0:
        for i in range((len(all_seen_rels)-1)//batch_size+1):
            seen_rels_batch = all_seen_rels[i*batch_size:(i+1)*batch_size]
            relations = [torch.tensor(all_relations[i],
                                          dtype=torch.long).to(device)
                             for i in seen_rels_batch]
            model.init_hidden(device, len(relations))
            ranked_relations, reverse_relation_indexs = \
                ranking_sequence(relations)
            relation_lengths = [len(relation) for relation in ranked_relations]
            #print(ranked_relations)
            pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
            new_rel_embeds = model.compute_que_embed(pad_relations, relation_lengths,
                                                 reverse_relation_indexs)
            for i, rel in enumerate(seen_rels_batch):
                rel_embeds[rel] = new_rel_embeds[i].cpu().numpy()
        rels = list(rel_embeds.keys())
        values = rel_embeds.values()
        with open(file_name, 'w') as writer:
            writer.write(str(rels)+'\n')
            for embed in values:
                to_write = [round(x, 6) for x in embed]
                writer.write(str(to_write)+'\n')

def update_rel_cands(memory_data, all_seen_cands, rel_embeds):
    if len(memory_data) >0:
        for this_memory in memory_data:
            for sample in this_memory:
                sample = [sample[0],
                          random.sample(all_seen_cands,
                                        min(num_cands,len(all_seen_cands))),
                          sample[2]]
                #sample = [sample[0], get_nearest_cand(sample[0], all_seen_cands,
                #                                      rel_embeds, num_cands)]

def get_que_embed(model, sample_list, all_relations):
    ret_que_embeds = []
    for i in range((len(sample_list)-1)//batch_size+1):
        samples = sample_list[i*batch_size:(i+1)*batch_size]
        questions = []
        for item in samples:
            this_question = torch.tensor(item[2], dtype=torch.long).to(device)
            questions.append(this_question)
        #print(len(questions))
        model.init_hidden(device, len(questions))
        ranked_questions, reverse_question_indexs = \
            ranking_sequence(questions)
        question_lengths = [len(question) for question in ranked_questions]
        #print(ranked_questions)
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        que_embeds = model.compute_que_embed(pad_questions, question_lengths,
                                             reverse_question_indexs)
        ret_que_embeds.append(que_embeds.detach().cpu().numpy())
    return np.concatenate(ret_que_embeds)

def select_data(model, samples, num_sel_data, all_relations):
    que_embeds = get_que_embed(model, samples, all_relations)
    #print(que_embeds[:5])
    num_clusters = min(num_sel_data, len(samples))
    distances = KMeans(n_clusters=num_clusters,
                    random_state=0).fit_transform(que_embeds)
    selected_samples = []
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        selected_samples.append(samples[sel_index])
    return selected_samples

def run_sequence(training_data, testing_data, valid_data, all_relations,
                 vocabulary,embedding, cluster_labels, num_clusters,
                 shuffle_index, rel_embeds):
    splited_training_data = split_data(training_data, cluster_labels,
                                       num_clusters, shuffle_index)
    splited_valid_data = split_data(valid_data, cluster_labels,
                                    num_clusters, shuffle_index)
    splited_test_data = split_data(testing_data, cluster_labels,
                                   num_clusters, shuffle_index)
    #print(splited_training_data)
    '''
    for data in splited_training_data[0]:
        print(data)
        print(cluster_labels[data[0]])
    '''
    #print(cluster_labels)
    seen_relations = []
    current_model = None
    memory_data = []
    sequence_results = []
    #np.set_printoptions(precision=3)
    result_whole_test = []
    relations_frequences_all = {}
    relations_frequences_task = []
    rel_ques_cand = {}
    rel_samples = {}
    full_rel_samples = {}
    rel_acc_diff = {}
    all_seen_rels = []
    past_fisher = None
    num_past_data = 0
    all_used_rels = list(rel_embeds.keys())
    for i in range(num_clusters):
        seen_relations += [data[0] for data in splited_training_data[i] if
                          data[0] not in seen_relations]
        current_train_data = remove_unseen_relation(splited_training_data[i],
                                                    seen_relations)
        current_valid_data = remove_unseen_relation(splited_valid_data[i],
                                                    seen_relations)
        current_test_data = []
        for j in range(i+1):
            current_test_data.append(
                remove_unseen_relation(splited_test_data[j], seen_relations))
        one_memory_data = []
        '''
        if i > 0:
            memory_data = sample_constrains(rel_samples,
                                            relations_frequences_all)
                                            '''
        '''
        for j in range(i):
            memory_data.append(sample_relations(relations_frequences_all,
                                                rel_ques_cand,
                                                task_memory_size,
                                                current_train_data))
        if i > 0:
            one_memory_data = sample_relations(relations_frequences_all,
                                           rel_ques_cand,
                                           len(current_train_data),
                                           current_train_data)
        '''
        #all_seen_rels = list(relations_frequences_all.keys())
        for this_sample in current_train_data:
            if this_sample[0] not in all_seen_rels:
                all_seen_rels.append(this_sample[0])
            for this_cand in this_sample[1]:
                if this_cand not in all_seen_rels:
                    all_seen_rels.append(this_cand)
        update_rel_embed(current_model, all_seen_rels, all_relations, rel_embeds)
        to_train_data = current_train_data+one_memory_data
        update_rel_cands(memory_data, all_seen_rels, rel_embeds)
        #random.shuffle(to_train_data)
        current_model, acc_diff = train(to_train_data, current_valid_data,
                              vocabulary, embedding_dim, hidden_dim,
                              device, batch_size, lr, model_path,
                              embedding, all_relations, current_model, epoch,
                              memory_data, loss_margin, past_fisher,
                              rel_samples, relations_frequences_all,
                              rel_embeds, rel_ques_cand, rel_acc_diff,
                                        all_seen_rels, update_rel_embed)
        #updata_saved_relations(current_train_data, rel_samples,
        #                       relations_frequences_all, rel_acc_diff, acc_diff)
        #updata_full_saved_relations(splited_training_data[i], full_rel_samples)
        #rel_samples = rm_unseen_rels(full_rel_samples, seen_relations)
        #save_rel_embeds(current_model, all_seen_rels, all_relations,
        #                'model_embed/embed'+str(i)+'.txt')
        #to_save_data = filter_data(current_train_data, current_model,
        #                           all_relations)
        #enlarge_rel_graph(current_train_data, None,
        #                   rel_ques_cand)
        '''
        past_fisher, num_past_data = update_fisher(current_model,
                                                   current_train_data,
                                                   all_relations,
                                                   past_fisher,
                                                   num_past_data)
                                                   '''
        #memory_data.append(current_train_data[-task_memory_size:])
        memory_data.append(select_data(current_model, current_train_data,
                                       task_memory_size, all_relations))
        results = [evaluate_model(current_model, test_data, batch_size,
                                  all_relations, device)
                   for test_data in current_test_data]
        print_list(results)
        sequence_results.append(np.array(results))
        result_whole_test.append(evaluate_model(current_model,
                                                testing_data, batch_size,
                                                all_relations, device))
    print('test set size:', [len(test_set) for test_set in current_test_data])
    #print('whole_test:', result_whole_test)
    return sequence_results, result_whole_test

def print_avg_results(all_results):
    avg_result = []
    for i in range(len(all_results[0])):
        avg_result.append(np.average([result[i] for result in all_results], 0))
    for line_result in avg_result:
        print_list(line_result)
    return avg_result

def print_avg_cand(sample_list):
    cand_lengths = []
    for sample in sample_list:
        cand_lengths.append(len(sample[1]))
    print('avg cand size:', np.average(cand_lengths))

if __name__ == '__main__':
    random_seed = int(sys.argv[1])
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    #bert_rel_features = compute_rel_embed(training_data)
    #print_avg_cand(training_data)
    cluster_labels, rel_features = cluster_data(num_clusters)
    to_use_embed = rel_features
    random.seed(random_seed)
    start_time = time.time()
    all_results = []
    result_all_test_data = []
    for i in range(sequence_times):
        shuffle_index = list(range(num_clusters))
        random_seed = int(sys.argv[1]) + 100*i
        random.seed(random_seed)
        #random.seed(random_seed+100*i)
        random.shuffle(shuffle_index)
        sequence_results, result_whole_test = run_sequence(
            training_data, testing_data, valid_data, all_relations,
            vocabulary, embedding, cluster_labels, num_clusters, shuffle_index,
            to_use_embed)
        all_results.append(sequence_results)
        result_all_test_data.append(result_whole_test)
    avg_result_all_test = np.average(result_all_test_data, 0)
    for result_whole_test in result_all_test_data:
        print_list(result_whole_test)
    print_list(avg_result_all_test)
    print_avg_results(all_results)
    end_time = time.time()
    #elapsed_time = end_time - start_time
    elapsed_time = (end_time - start_time) / sequence_times
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
