
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import CONFIG as conf

# build word-relation tensor
def build_word_relation(word_relation, device):
    return [torch.tensor(word_relation[0], dtype=torch.long).to(device),
            torch.tensor(word_relation[1], dtype=torch.long).to(device)]

# process the data by adding questions
def process_testing_samples(sample_list, all_relations, device):
    questions = []
    word_relations = []
    gold_relation_indexs = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        gold_relation_indexs.append(sample[0])
        neg_word_relation = [build_word_relation(all_relations[index], device)
                             for index in sample[1]]
        relation_set_lengths.append(len(neg_word_relation))
        word_relations += neg_word_relation
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return gold_relation_indexs, questions, word_relations, \
        relation_set_lengths

# process the data by adding questions
def process_samples(sample_list, all_relations, device):
    questions = []
    word_relations = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        pos_word_relation = build_word_relation(all_relations[sample[0]], device)
        #print(sample[1])
        #print(len(all_relations))
        neg_word_relation = [build_word_relation(all_relations[index], device)
                             for index in sample[1]]
        relation_set_lengths.append(len(neg_word_relation)+1)
        word_relations += [pos_word_relation]+ neg_word_relation
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return questions, word_relations, relation_set_lengths

def ranking_sequence(sequence):
    word_lengths = torch.tensor([len(sentence) for sentence in sequence])
    rankedi_word, indexs = word_lengths.sort(descending = True)
    ranked_indexs, inverse_indexs = indexs.sort()
    #print(indexs)
    sequence = [sequence[i] for i in indexs]
    return sequence, inverse_indexs

def ranking_word_relation(sequence):
    word_lengths = torch.tensor([len(sentence[0])+len(sentence[1]) for sentence in sequence])
    rankedi_word, indexs = word_lengths.sort(descending = True)
    ranked_indexs, inverse_indexs = indexs.sort()
    #print(indexs)
    sequence = [sequence[i] for i in indexs]
    return sequence, inverse_indexs
