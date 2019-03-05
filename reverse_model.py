'''
This code is based on the Pytorch Orientaion:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
Original Author: Robert Guthrie
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import CONFIG as conf

torch.manual_seed(1)
hidden_dim = conf['hidden_dim']
l_rate = conf['lr_revers_model']
epochs = conf['epoch_revers_model']
device = conf['device']

class ReverseModel(nn.Module):
    def __init__(self, input_dim, output_dim):

            super(ReverseModel, self).__init__()
            # Calling Super Class's constructor
            self.linear = nn.Linear(input_dim, output_dim)
            # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out

def feed_samples(reverse_model, x_train, y_correct, criterion, optimiser):
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_correct).to(device)
    optimiser.zero_grad()
    outputs = reverse_model.forward(inputs)
    #loss = torch.mean(criterion(outputs, labels))
    loss = criterion(outputs, labels)
    #print(loss)
    loss.backward()# back props
    optimiser.step()

def update_reverse_model(reverse_model, cur_que_embed, cur_rel_embed,
                         memory_que_embed, memory_rel_embed):
    if reverse_model is None:
        reverse_model = ReverseModel(hidden_dim*2, hidden_dim*2)
    reverse_model = reverse_model.to(device)
    criterion = nn.MSELoss()
    #criterion = nn.CosineSimilarity(dim=1)
    #l_rate = 0.0001
    optimiser = torch.optim.Adam(reverse_model.parameters(), lr = l_rate)
    #epochs = 10
    for epoch in range(epochs):
        for i in range(len(cur_que_embed)):
            que_x = cur_que_embed[i]
            que_y = memory_que_embed[i]
            feed_samples(reverse_model, que_x, que_y, criterion, optimiser)
            rel_x = cur_rel_embed[i]
            rel_y = memory_rel_embed[i]
            feed_samples(reverse_model, rel_x, rel_y, criterion, optimiser)
    return reverse_model
