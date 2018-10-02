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
hidden_dim = 100
batch_size = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# evaluate the model on the testing data
def evaluate_model(model, testing_data):
    num_correct = 0
    for sample in testing_data[0:10]:
        question = torch.tensor(sample[2], dtype=torch.long)
        gold_relation_index = sample[0]
        cand_relations = [torch.tensor(relations[index], dtype=torch.long)
                         for index in sample[1]]
        cand_scores = []
        for relation in cand_relations:
            model.init_hidden()
            cand_scores.append(model(question, relation))
        pred_index = sample[1][np.argmax(cand_scores)]
        if pred_index == gold_relation_index:
            num_correct += 1
    #print(cand_scores[-1])
    print('num correct:', num_correct)
    print('correct rate:', num_correct/len(testing_data))

def process_samples(sample_list):
    all_questions = []
    all_relations = []
    relation_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long)
        #print(relations[sample[0]])
        print(sample)
        print(len(relations))
        pos_relation = torch.tensor(relations[sample[0]], dtype=torch.long)
        neg_relations = [torch.tensor(relations[index], dtype=torch.long)
                         for index in sample[1]]
        relation_lengths.append(len(neg_relations)+1)
        all_relations += [pos_relation]+ neg_relations
        #all_questions += [question for i in range(relation_lengths[-1])]
        all_questions += [question] * relation_lengths[-1]
    return all_questions, all_relations, relation_lengths

if __name__ == '__main__':
    training_data, testing_data, valid_data, relations, vocabulary,  embedding=\
        gen_data()
    #print(training_data[0:10])
    #print(testing_data[0:10])
    #print(valid_data[0:10])
    #print(relations[0:10])
    model = SimilarityModel(embedding_dim, hidden_dim, len(vocabulary),
                            np.array(embedding), 3)
    loss_function = nn.MarginRankingLoss(0)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.to(device)
    for epoch in range(100):
        print('epoch', epoch)
        for i in range((len(training_data)-1)//batch_size+1):
            samples = training_data[i*batch_size:(i+1)*batch_size]
            questions, relations, relation_lengths = process_samples(samples)
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

            model.init_hidden(sum(relation_lengths))
            print(sum(relation_lengths))
            all_scores = model(questions, relations, device)
            print(all_scores)
            #print(relation_lengths)
            pos_scores = []
            neg_scores = []
            start_index = 0
            for length in relation_lengths:
                pos_scores.append(all_scores[start_index].expand(length-1))
                neg_scores.append(all_scores[start_index+1:start_index+length])
                start_index += length
            pos_scores = torch.cat(pos_scores)
            neg_scores = torch.cat(neg_scores)

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
            #print(neg_score)
            loss = loss_function(pos_scores, neg_scores,
                                 torch.ones(sum(relation_lengths)-len(relation_lengths)))
            loss.backward(retain_graph=True)
            #print('loss', loss)
            optimizer.step()
        evaluate_model(model, testing_data)
