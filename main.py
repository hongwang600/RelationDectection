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
    print('correct rate:', num_correct/len(testing_data))

if __name__ == '__main__':
    training_data, testing_data, valid_data, relations, vocabulary,  embedding=\
        gen_data()
    #print(training_data[0:10])
    #print(testing_data[0:10])
    #print(valid_data[0:10])
    #print(relations[0:10])
    model = SimilarityModel(embedding_dim, hidden_dim, len(vocabulary),
                            np.array(embedding))
    loss_function = nn.MarginRankingLoss(0)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(100):
        print('epoch', epoch)
        for sample in training_data[0:10]:
            model.zero_grad()
            question = torch.tensor(sample[2], dtype=torch.long)
            #print(relations[sample[0]])
            pos_relation = torch.tensor(relations[sample[0]], dtype=torch.long)
            neg_relations = [torch.tensor(relations[index], dtype=torch.long)
                             for index in sample[1]]
            if len(neg_relations) > 0:
                model.init_hidden()
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
                #print(neg_score)
                loss = loss_function(pos_score, neg_score,
                                     torch.ones(len(neg_relations)))
                loss.backward(retain_graph=True)
                #print('loss', loss)
                optimizer.step()
        evaluate_model(model, testing_data)
