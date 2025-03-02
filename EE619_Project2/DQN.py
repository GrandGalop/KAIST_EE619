
import numpy as np
import torch
import pdb
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque

np.random.seed(1)

class DeepQLearning:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,
            discount_factor,
            e_greedy,
            replace_target_iter,
            memory_size,
            batch_size
        ):
        ############ To DO #################
        # Initialize variables
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e_greedy = e_greedy
        self.eps = 0.5
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        self.memory = deque()
        #self.criterion = nn.MSELoss()
        self.construct_network()
        self.target_network = self.model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.numiter = 0


    def construct_network(self):
        ############# To Do ##############
        self.model = nn.Sequential(
            nn.Linear(self.n_features, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(0.5),                       
            nn.Linear(8, self.n_actions))
    
    def store_transition(self, s, a, r, next_s):
        ############ To Do #############
        e = (s, a, r, next_s)
        self.memory.append(e)
    
    def epsilon_decay(self):
        self.eps = max(self.eps*0.9, self.e_greedy)

    def choose_action(self, state):
        ############# To Do ##############
        state = torch.tensor(state, dtype=torch.float32)
        #print(state)

        qvalues = self.model(state).detach().numpy()
        #print(qvalues)

        if state[0] + 0.9 ==0 :
            qvalues[2] = float("-inf")
        if state[0] + 0.0 ==0 :
            qvalues[3] = float("-inf")
        if state[1] - 0.9 ==0 :
            qvalues[1] = float("-inf")            
        if state[1] + 0.0 ==0 :
            qvalues[0] = float("-inf")
        #print(qvalues)

        best_action = np.argmax(qvalues)
        probability_array = []
        for index in range(len(qvalues)):
            if index==int(best_action):
                probability_array.append(1 - self.eps + (self.eps / len(qvalues)))
            else:
                probability_array.append(self.eps / len(qvalues))

        self.numiter +=1
        if self.numiter % 5000==0:
            self.epsilon_decay()
        #print(probability_array)
        action = np.random.choice([0, 1, 2, 3], 1, p=probability_array).item()
        #print(action)
        return action

    def learn(self):
        ############# To Do ##############
        samplenum = np.random.choice(len(self.memory),min(len(self.memory),self.batch_size), replace=False)
        batch = [self.memory[n]for n in samplenum]
        for element in batch:
            state, action, reward, next_s = element
            state = torch.tensor(state, dtype=torch.float32)
            next_s = torch.tensor(next_s, dtype=torch.float32)
            loss = (reward + max(self.target_network(next_s)) - max(self.model(state)))**2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.numiter % self.replace_target_iter == 0:
            self.target_network.load_state_dict(self.model.state_dict())

# dqn = DeepQLearning(4, 2, 0.003, 4, 0.3, 0.3, 7, 3)
# a = np.array([0, 1])
# b = torch.tensor([0, 1], dtype = torch.float32)
# print(dqn.choose_action(a))
# print(dqn.model(b))
# print(dqn.choose_action(b))
# dqn.store_transition(np.array([0, 1]), 2, 3, np.array([1, 1]))
# dqn.learn()
# dqn.learn()
# dqn.learn()
# dqn.learn()
# dqn.learn()
# dqn.learn()
# dqn.learn()
# dqn.learn()
# dqn.learn()
# dqn.learn()
# dqn.learn()
