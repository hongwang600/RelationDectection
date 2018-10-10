import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence,\
    ranking_word_relation
from config import CONFIG as conf

model_path = conf['model_path']
batch_size = conf['batch_size']
device = conf['device']

# evaluate the model on the testing data
def evaluate_model(model, testing_data, batch_size, all_relations, device):
    #print('start evaluate')
    num_correct = 0
    #testing_data = testing_data[0:100]
    for i in range((len(testing_data)-1)//batch_size+1):
        samples = testing_data[i*batch_size:(i+1)*batch_size]
        gold_relation_indexs, questions, word_relations, relation_set_lengths =\
        process_testing_samples(samples, all_relations, device)
        ranked_questions, reverse_question_indexs = \
            ranking_sequence(questions)
        question_lengths = [len(question) for question in ranked_questions]
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        pad_questions = pad_questions.to(device)
        ranked_word_relations, reverse_word_relation_indexs =\
            ranking_word_relation(word_relations)
        word_relation_lengths = [len(word_relation[0])+len(word_relation[1])
                                 for word_relation in ranked_word_relations]
        #print(pad_questions)

        model.zero_grad()
        model.init_hidden(device, sum(relation_set_lengths))
        all_scores = model(pad_questions,
                           reverse_question_indexs,
                           question_lengths,
                           ranked_word_relations,
                           reverse_word_relation_indexs,
                           word_relation_lengths,
                           device)
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
    #print('num correct:', num_correct)
    #print('correct rate:', float(num_correct)/len(testing_data))
    return float(num_correct)/len(testing_data)

if __name__ == '__main__':
    model = torch.load(model_path)
    training_data, testing_data, valid_data, all_relations, word_vocabulary, \
        word_embedding, relation_vocabulary, relation_embeddeing = gen_data()
    #model.init_embedding(np.array(embedding))
    acc=evaluate_model(model, testing_data, batch_size, all_relations, device)
    print('accuracy:', acc)
