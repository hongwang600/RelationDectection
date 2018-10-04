
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# process the data by adding questions
def process_testing_samples(sample_list, all_relations, device):
    questions = []
    relations = []
    gold_relation_indexs = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        gold_relation_indexs.append(sample[0])
        neg_relations = [torch.tensor(all_relations[index],
                                      dtype=torch.long).to(device)
                         for index in sample[1]]
        relation_set_lengths.append(len(neg_relations))
        relations += neg_relations
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return gold_relation_indexs, questions, relations, relation_set_lengths

# evaluate the model on the testing data
def evaluate_model(model, testing_data, batch_size, all_relations, device):
    print('start evaluate')
    num_correct = 0
    #testing_data = testing_data[0:100]
    for i in range((len(testing_data)-1)//batch_size+1):
        samples = testing_data[i*batch_size:(i+1)*batch_size]
        gold_relation_indexs, questions, relations, relation_set_lengths = \
            process_testing_samples(samples, all_relations, device)
        model.init_hidden(device, sum(relation_set_lengths))
        ranked_questions, reverse_question_indexs = \
            ranking_sequence(questions)
        ranked_relations, reverse_relation_indexs =\
            ranking_sequence(relations)
        question_lengths = [len(question) for question in ranked_questions]
        relation_lengths = [len(relation) for relation in ranked_relations]
        #print(ranked_questions)
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
        all_scores = model(pad_questions, pad_relations, device,
                           reverse_question_indexs, reverse_relation_indexs,
                           question_lengths, relation_lengths)
        start_index = 0
        pred_indexs = []
        #print('len of relation_set:', len(relation_set_lengths))
        for j in range(len(relation_set_lengths)):
            length = relation_set_lengths[j]
            cand_indexs = samples[j][1]
            pred_index = (cand_indexs[
                all_scores[start_index:start_index+length].argmax()])
            if pred_index == gold_relation_indexs[j]:
                num_correct += 1
            #print('scores:', all_scores[start_index:start_index+length])
            #print('cand indexs:', cand_indexs)
            #print('pred, true:',pred_index, gold_relation_indexs[j])
            start_index += length
    #print(cand_scores[-1])
    print('num correct:', num_correct)
    print('correct rate:', float(num_correct)/len(testing_data))
    return float(num_correct)/len(testing_data)

# process the data by adding questions
def process_samples(sample_list, all_relations, device):
    questions = []
    relations = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        pos_relation = torch.tensor(all_relations[sample[0]],
                                    dtype=torch.long).to(device)
        neg_relations = [torch.tensor(all_relations[index],
                                      dtype=torch.long).to(device)
                         for index in sample[1]]
        relation_set_lengths.append(len(neg_relations)+1)
        relations += [pos_relation]+ neg_relations
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return questions, relations, relation_set_lengths

def ranking_sequence(sequence):
    word_lengths = torch.tensor([len(sentence) for sentence in sequence])
    rankedi_word, indexs = word_lengths.sort(descending = True)
    ranked_indexs, inverse_indexs = indexs.sort()
    #print(indexs)
    sequence = [sequence[i] for i in indexs]
    return sequence, inverse_indexs
