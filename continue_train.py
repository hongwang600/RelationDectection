import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import time

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence
from evaluate import evaluate_model
from data_partition import cluster_data
from config import CONFIG as conf
from train import train

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

def print_list(result):
    for num in result:
        sys.stdout.write('%.3f, ' %num)
    print('')


def enlarge_rel_graph(train_data, relations_frequences, rel_ques_cand):
    #new_relation_frequences = {}
    for sample in train_data:
        pos_index = sample[0]
        neg_cands = sample[1]
        question = sample[2]
        if pos_index not in relations_frequences:
            relations_frequences[pos_index] = 1
        else:
            relations_frequences[pos_index] += 1
            relations_frequences[pos_index] = \
                max(50, relations_frequences[pos_index])
        if pos_index not in rel_ques_cand:
            rel_ques_cand[pos_index] = [neg_cands, [question]]
        else:
            if len(rel_ques_cand[pos_index][1]) < 10:
                rel_ques_cand[pos_index][1].append(question)
            for cand_rel in neg_cands:
                if cand_rel not in rel_ques_cand[pos_index][0]:
                    rel_ques_cand[pos_index][0].append(cand_rel)
        all_rels = [pos_index] + neg_cands
        for cand_rel in neg_cands:
            if cand_rel in rel_ques_cand:
                rel_ques_cand[cand_rel][0] += [rel for rel in all_rels if rel
                                               not in rel_ques_cand[cand_rel][0]
                                               and rel != cand_rel]
            else:
                rel_ques_cand[cand_rel] = [[rel for rel in all_rels if
                                            rel!=cand_rel], []]

def random_walk(rel_ques_cand, num_cands, rel):
    cand_set = []
    cur_rel = rel
    cur_num_cands = 0
    while cur_num_cands < num_cands:
        cur_rel = random.sample(rel_ques_cand[cur_rel][0], 1)[0]
        if cur_rel != rel:
            cand_set.append(cur_rel)
            cur_num_cands += 1
    return cand_set
    #all_cands = list(rel_ques_cand.keys())
    #return random.sample(all_cands, num_cands)

def sample_relations(relations_frequences, rel_ques_cand, num_samples):
    relations = list(relations_frequences.keys())
    relation_fre = np.array(list(relations_frequences.values()))
    relation_pro = relation_fre/float(sum(relation_fre))
    selected_rels = np.random.choice(relations, num_samples, True, relation_pro)
    ret_relations = []
    for rel in selected_rels:
        #print(rel_ques_cand[rel])
        question = random.sample(rel_ques_cand[rel][1],1)[0]
        neg_cands = random_walk(rel_ques_cand, num_cands, rel)
        ret_relations.append([rel, neg_cands, question])
    #print(ret_relations)
    return ret_relations

def sample_relations_task(relations_frequences, rel_ques_cand, num_samples):
    ret_samples = []
    for relation_fre in relations_frequences:
        task_sample = sample_relations(relation_fre, rel_ques_cand, num_samples)
        ret_samples.append(task_sample)
    return ret_samples

def run_sequence(training_data, testing_data, valid_data, all_relations,
                 vocabulary,embedding, cluster_labels, num_clusters,
                 shuffle_index):
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
        current_model = train(current_train_data, current_valid_data,
                              vocabulary, embedding_dim, hidden_dim,
                              device, batch_size, lr, model_path,
                              embedding, all_relations, current_model, epoch,
                              memory_data, loss_margin)
        '''
        enlarge_rel_graph(current_train_data, relations_frequences_all,
                          rel_ques_cand)
        memory_data = [sample_relations(relations_frequences_all, rel_ques_cand,
                                        task_memory_size*(i+1))]
                                        '''
        #'''
        new_rel_frequences = {}
        enlarge_rel_graph(current_train_data, new_rel_frequences,
                          rel_ques_cand)
        relations_frequences_task.append(new_rel_frequences)
        memory_data = sample_relations_task(relations_frequences_task,
                                            rel_ques_cand,
                                            task_memory_size)
        #                                    '''
        #memory_data.append(current_train_data[-task_memory_size:])
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
    #print_avg_cand(training_data)
    cluster_labels = cluster_data(num_clusters)
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
            vocabulary, embedding, cluster_labels, num_clusters, shuffle_index)
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
