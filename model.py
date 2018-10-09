'''
This code is based on the Pytorch Orientaion:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
Original Author: Robert Guthrie
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim,
                 batch_size, device):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        #self.maxpool = nn.MaxPool1d(hidden_dim*2)
        self.hidden = self.build_hidden()

    def build_hidden(self, batch_size = 1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #print(batch_size)
        return [torch.zeros(2, batch_size, self.hidden_dim),
                torch.zeros(2, batch_size, self.hidden_dim)]

    def init_hidden(self, device='cpu', batch_size = 1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #print(batch_size)
        self.hidden = (torch.zeros(2, batch_size, self.hidden_dim).to(device),
                torch.zeros(2, batch_size, self.hidden_dim).to(device))
        #self.hidden[0] = self.hidden[0].to(device)
        #self.hidden[1] = self.hidden[1].to(device)

    def forward(self, packed_embeds):
        #print(packed_embeds)
        #print(self.hidden)
        lstm_out, self.hidden = self.lstm(packed_embeds, self.hidden)
        #maxpool_hidden = self.maxpool(lstm_out.view(1,len(sentence), -1))
        permuted_hidden = self.hidden[0].permute([1,0,2]).contiguous()
        #print(permuted_hidden.size())
        return permuted_hidden.view(-1, self.hidden_dim*2)

class SimilarityModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,
                 word_vocab_size, relation_vocab_size, word_vocab_embedding,
                 relation_vocab_embedding,
                 batch_size, device):
        super(SimilarityModel, self).__init__()
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(word_vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(word_vocab_embedding))
        self.word_embeddings = self.word_embeddings.to(device)
        self.word_embeddings.weight.requires_grad = False
        self.relation_embeddings = nn.Embedding(
            relation_vocab_size, embedding_dim)
        self.relation_embeddings.weight.data.copy_(
            torch.from_numpy(relation_vocab_embedding))
        self.relation_embeddings = self.relation_embeddings.to(device)
        #self.relation_embeddings.weight.requires_grad = False
        self.sentence_biLstm = BiLSTM(embedding_dim, hidden_dim,
                                      batch_size, device)
        self.relation_biLstm = BiLSTM(embedding_dim, hidden_dim,
                                      batch_size, device)

    def init_hidden(self, device, batch_size=1):
        self.sentence_biLstm.init_hidden(device, batch_size)
        self.relation_biLstm.init_hidden(device, batch_size)

    def init_embedding(self, vocab_embedding):
        #print(self.word_embeddings(torch.tensor([27]).cuda()))
        self.word_embeddings.weight.data.copy_(torch.from_numpy(vocab_embedding))
        #print(self.word_embeddings(torch.tensor([27]).cuda()))

    def ranking_sequence(self, sequence):
        word_lengths = torch.tensor([len(sentence) for sentence in sequence])
        rankedi_word, indexs = word_lengths.sort(descending = True)
        ranked_indexs, inverse_indexs = indexs.sort()
        #print(indexs)
        sequence = [sequence[i] for i in indexs]
        return sequence, inverse_indexs

    def get_word_relation_embedding(self, word_relation_list):
        return [torch.cat([self.word_embeddings(word_relation[0]),
                           self.relation_embeddings(word_relation[1])]) for
                word_relation in word_relation_list]

    def forward(self, question_list,reverse_question_indexs, question_lengths,
                word_relation_list, reverse_word_relation_indexs,
                word_relation_lengths, device):
        '''
        question_embeds = [self.word_embeddings(sentence)
                           for sentence in question_list]
        ranked_question_embeds, reverse_question_indexs = \
            self.ranking_sequence(question_embeds)
        #print(question_embeds)
        question_packed = torch.nn.utils.rnn.pack_sequence(ranked_question_embeds)
        relation_embeds = [self.word_embeddings(sentence)
                           for sentence in relation_list]
        ranked_relation_embeds, reverse_relation_indexs =\
            self.ranking_sequence(relation_embeds)
        relation_packed = torch.nn.utils.rnn.pack_sequence(ranked_relation_embeds)
        question_packed.to(device)
        relation_packed.to(device)
        '''
        question_embeds = self.word_embeddings(question_list)
        word_relation_embeddings =\
            self.get_word_relation_embedding(word_relation_list)

        #print(question_lengths)
        question_packed = \
            torch.nn.utils.rnn.pack_padded_sequence(question_embeds,
                                                    question_lengths)
        word_relation_packed = \
            torch.nn.utils.rnn.pack_sequence(word_relation_embeddings)
        #question_packed.to(device)
        #relation_packed.to(device)
        #self.sentence_biLstm.to(device)
        #print(question_packed)
        #print(question_lengths)
        question_lstm_embedding = self.sentence_biLstm(question_packed)
        relation_lstm_embedding = self.relation_biLstm(word_relation_packed)
        question_lstm_embedding = question_lstm_embedding[reverse_question_indexs]
        relation_lstm_embedding = \
            relation_lstm_embedding[reverse_word_relation_indexs]
        #print('sentence_embedding size', sentence_embedding.size())
        #print('relation_embedding size', relation_embedding.size())
        #print('sentence_embedding', sentence_embedding)
        #print('relation_embedding', relation_embedding)
        cos = nn.CosineSimilarity(dim=1)
        return cos(question_lstm_embedding, relation_lstm_embedding)
