
import numpy as np
import torch
import pdb
import random
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


np.random.seed(1)

class DNN(nn.Module) :
    def __init__(self, n_actions, n_features) :
        super(DNN, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.model = nn.Sequential(
            nn.Linear(self.n_features, 4),
            nn.ReLU(),
            #nn.Dropout(0.25),
            nn.Linear(4, 8),
            nn.ReLU(),
            #nn.Dropout(0.25),

            nn.Linear(8, n_actions),
            )
        self._init_weights()

    def _init_weights(self):

        for module in self.modules() :
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                #nn.init.orthogonal(module.weight)
                #nn.init.kaiming_uniform_(module.weight)

                #module.weight.data.normal_(mean=0.0, std=1.0)
                if module.bias is not None:
                    module.bias.data.zero_()

    
    def forward(self, x) :
        return self.model(x)



class Reinforce:
    def __init__(self, n_actions, n_features, learning_rate, discount_factor, eps):

        ############ To DO #################
        # Initialize variables
        super(Reinforce, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps = eps
        self.construct_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        

    def construct_network(self):
        ############ To DO #################
        self.model = DNN(self.n_actions, self.n_features)
    
    
    def choose_action(self, state):
        ############ To DO #################
        state = torch.tensor(state, dtype=torch.float32)
        probability_array = F.softmax(self.model(state))
        #print(probability_array)
        # if state[0] + 0.9 ==0 :
        #     probability_array[2] = 0
        # if state[0] + 0.0 ==0 :
        #     probability_array[3] = 0
        # if state[1] - 0.9 ==0 :
        #     probability_array[1] = 0           
        # if state[1] + 0.0 ==0 :
        #     probability_array[0] = 0

        #probability_array = torch.clamp(probability_array, max=0.9, min=0.05)
        # if random.random() < 1-self.eps:
        #     action = torch.multinomial(torch.tensor([0.25, 0.25, 0.25, 0.25]), 1).item()
        # else:
        action = torch.multinomial(probability_array, 1).item()
        return action, probability_array[action]

    # def eps_decay(self):
    #     self.epsilon = max(self.epsilon * 0.9, self.eps)

    def learn(self): #reward_array, log_prob_array):
        ############ To DO #################
        returns = []
        discounted_reward = 0
        for reward in reversed(self.saved_rewards):
            discounted_reward = reward + discounted_reward * self.discount_factor
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)
        #print(returns)

        log_probs = torch.stack(self.saved_log_probs)
        #print(log_probs)
        loss = -torch.mean(returns * log_probs)
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(),5)
        self.optimizer.step()


dqn = Reinforce(4, 2, 0.001, 0.9, 0.1)
a = np.array([0, 1])
b = torch.tensor([0, 1], dtype = torch.float32)
print(dqn.choose_action(a))
# # print(dqn.model(b))
# print(dqn.choose_action(b))