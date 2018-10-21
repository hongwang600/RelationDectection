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
from utils import process_testing_samples, process_samples, ranking_sequence
from evaluate import evaluate_model, compute_diff_scores
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

def select_samples(model, samples, task_memory_size, all_relations,
                   relation_embeding_set):
    diff_scores = compute_diff_scores(model, samples, batch_size, all_relations,
                                      device)
    distinct_relations = []
    for this_sample in samples:
        if this_sample[0] not in distinct_relations:
            distinct_relations.append(this_sample[0])
    #print(distinct_relations)
    distinct_embeds = [relation_embeding_set[i] for i in distinct_relations]
    #num_clusters = min(task_memory_size, len(distinct_embeds))
    num_clusters = min(10, len(distinct_embeds))
    kmeans = KMeans(n_clusters=num_clusters,
                    random_state=0).fit(distinct_embeds)
    labels = kmeans.labels_
    cluster_index = {}
    for i in range(len(distinct_relations)):
        cluster_index[distinct_relations[i]] = labels[i]
    sample_in_cluster = [0 for i in range(num_clusters)]
    selected_samples = []
    total_num_samples = 0
    for this_sample in samples:
        cluster_label = cluster_index[this_sample[0]]
        if sample_in_cluster[cluster_label] < 1:
            selected_samples.append(this_sample)
            sample_in_cluster[cluster_label] += 1
            total_num_samples += 1
        if total_num_samples == num_clusters:
            break
    selected_samples += random.sample(samples, min(task_memory_size-num_clusters,
                                                 len(samples)))
    return selected_samples

def run_sequence(training_data, testing_data, valid_data, all_relations,
                 vocabulary,embedding, cluster_labels, num_clusters,
                 shuffle_index, relation_embeding_set):
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
        memory_data.append(select_samples(current_model, current_train_data,
                                          task_memory_size, all_relations,
                                          relation_embeding_set))
        results = [evaluate_model(current_model, test_data, batch_size,
                                  all_relations, device)
                   for test_data in current_test_data]
        print_list(results)
        sequence_results.append(np.array(results))
    return sequence_results

def print_avg_results(all_results):
    avg_result = []
    for i in range(len(all_results[0])):
        avg_result.append(np.average([result[i] for result in all_results], 0))
    for line_result in avg_result:
        print_list(line_result)
    return avg_result

if __name__ == '__main__':
    random_seed = int(sys.argv[1])
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    cluster_labels, relation_embeding_set = cluster_data(num_clusters)
    shuffle_index = [i for i in range(num_clusters)]
    random.seed(random_seed)
    start_time = time.time()
    all_results = []
    for i in range(sequence_times):
        random.seed(random_seed+100*i)
        random.shuffle(shuffle_index)
        all_results.append(run_sequence(training_data, testing_data,
                                        valid_data, all_relations,
                                        vocabulary, embedding, cluster_labels,
                                        num_clusters, shuffle_index,
                                        relation_embeding_set))
    print_avg_results(all_results)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / sequence_times
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
