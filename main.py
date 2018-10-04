'''
relation_file = "./data/relation.2M.list"
training_file = "./data/train.replace_ne.withpool"
test_file = "./data/test.replace_ne.withpool"
valid_file = "./data/valid.replace_ne.withpool"
#glove_file = "./data/glove.6B.300d.txt"
glove_file = "./data/small_glove.6B.300d.txt"
embedding_size = 300
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import gen_data
from model import SimilarityModel
embedding_dim = 300
hidden_dim = 200
batch_size = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

if __name__ == '__main__':
    training_data, testing_data, valid_data, all_relations, vocabulary,  embedding=\
        gen_data()
    #print(training_data[0:10])
    #print(testing_data[0:10])
    #print(valid_data[0:10])
    #print(all_relations[0:10])
    model = SimilarityModel(embedding_dim, hidden_dim, len(vocabulary),
                            np.array(embedding), 1, device)
    loss_function = nn.MarginRankingLoss(0.5)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1000):
        print('epoch', epoch)
        #training_data = training_data[0:100]
        for i in range((len(training_data)-1)//batch_size+1):
            samples = training_data[i*batch_size:(i+1)*batch_size]
            questions, relations, relation_set_lengths = process_samples(samples,
                                                                     all_relations,
                                                                     device)
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
            '''
            question = torch.tensor(sample[2], dtype=torch.long)
            #print(relations[sample[0]])
            pos_relation = torch.tensor(relations[sample[0]], dtype=torch.long)
            neg_relations = [torch.tensor(relations[index], dtype=torch.long)
                             for index in sample[1]]
            all_relations = [pos_relation]+ neg_relations
            expanded_question = question.expand(len(all_relations), -1)
            '''
            #print(expanded_question)

            # put data to device
            #expanded_question = expanded_question.to(device)
            #pos_relation = pos_relation.to(device)
            #neg_relations = [relation.to(device) for relation in neg_relations]
            #all_relations = all_relations.to(device)

            model.init_hidden(device, sum(relation_set_lengths))
            #print(sum(relation_set_lengths))
            #print(model)
            all_scores = model(pad_questions, pad_relations, device,
                               reverse_question_indexs, reverse_relation_indexs,
                               question_lengths, relation_lengths)
            #print(all_scores)
            #print(relation_set_lengths)
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
            #print('got scores')

            '''
            pos_score = model(question, pos_relation)
            pos_score_set = pos_score
            for i in range(len(neg_relations)-1):
                pos_score_set = torch.cat((pos_score_set, pos_score), 0)
            neg_score = []
            for relation in neg_relations:
                model.init_hidden()
                neg_score.append(model(question, relation))
            #print(pos_score)
            #print(neg_score)
            neg_score = torch.cat(neg_score, 0)
            '''
            #print(pos_scores)
            #print(neg_scores)
            loss = loss_function(pos_scores, neg_scores,
                                 torch.ones(sum(relation_set_lengths)-
                                            len(relation_set_lengths)))
            #print(loss)
            #loss.backward(retain_graph=True)
            loss.backward()
            #print('got loss')
            #print('loss', loss)
            optimizer.step()
        evaluate_model(model, valid_data, batch_size, all_relations, device)
