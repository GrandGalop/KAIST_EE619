import numpy as np
from collections import defaultdict

class Q_learning():
    def __init__(self, environment, epsilon=0.05, alpha=0.01, gamma=0.99):
        self.Q_table = defaultdict(lambda: np.zeros(4)) # Q_table = {"state":(Q(s,UP),Q(s,DOWN),Q(s,LEFT),Q(s,RIGHT),)}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = environment.actions # ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.env_row_max = environment.row_max
        self.env_col_max = environment.col_max
        
    def get_Q_table(self):
        return self.Q_table

    def action(self, state):
        '''
        In this code, you have to implement the behavior policy (epsilon-greedy policy) w.r.t. the Q-table.
        The policy takes a state and then samples an action among  ['UP', 'DOWN', 'LEFT', 'RIGHT'],
        and you can index the above actions as [0, 1, 2, 3]. Use "self.epsilon" and "self.Q_table".
        '''

        '''
        
        your codes here
        
        '''
        Q_table = self.get_Q_table()
        max_index = np.argmax(Q_table[state])
        probability = np.zeros(4)
        for i in range(0, len(probability)):
            if i == max_index:
                probability[i] = 1 - self.epsilon + self.epsilon/len(probability)
            else:
                probability[i] = self.epsilon/len(probability)
        action_index = np.random.choice(4, 1, p=probability).item()
        return self.actions[action_index]

    def update(self, current_state, next_state, action, reward):
        '''
        In this code, you should implement Q-learning update rule.
        '''

        '''

        your codes here

        '''
        action_index = {"UP": 0, "DOWN" : 1, "LEFT" : 2, "RIGHT" : 3}
        self.Q_table[current_state][action_index[action]] = (self.Q_table[current_state][action_index[action]] + 
                                               self.alpha * (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[current_state][action_index[action]]))


    def get_max_Q_function(self):
        '''
        This code gives max_a Q(s,a) for each state to us. The output of this code should be a form of "list".
        Therefore, the output "max_Q_table = [max_a Q(s,a)] = [max_a Q((row_index, col_index),a)]",
         and you already found the index of state "s" in GridWorld.py.
        '''
        max_Q_table = np.zeros((self.env_row_max, self.env_col_max))

        '''

        your codes here

        '''
        Q_table= self.get_Q_table()
        for i in range(0, self.env_row_max):
            for j in range(0, self.env_col_max):
                max_Q_table[i][j] = np.max(Q_table[(i, j)])

        return max_Q_table

class Double_Q_learning():
    def __init__(self, environment, epsilon=0.05, alpha=0.01, gamma=0.99):
        self.Q1 = defaultdict(lambda: np.zeros(4)) # Q_table = {"state":(Q(s,UP),Q(s,DOWN),Q(s,LEFT),Q(s,RIGHT),)}
        self.Q2 = defaultdict(lambda: np.zeros(4))
        self.Q_table = defaultdict(lambda: np.zeros(4))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = environment.actions # ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.env_row_max = environment.row_max
        self.env_col_max = environment.col_max
        
    def get_Q_table(self):
        return self.Q_table

    def action(self, state):
        '''

        your codes here
        you have to implement the behavior policy (epsilon-greedy policy) w.r.t. the Q-table.

        '''
        Q_table = self.get_Q_table()
        max_index = np.argmax(Q_table[state])
        probability = np.zeros(4)
        for i in range(0, len(probability)):
            if i == max_index:
                probability[i] = 1 - self.epsilon + self.epsilon/len(probability)
            else:
                probability[i] = self.epsilon/len(probability)
        action_index = np.random.choice(4, 1, p=probability).item()

        return self.actions[action_index]

    def update(self, current_state, next_state, action, reward):
        '''

        your codes here
        This code should contain the Double Q-learning update rule.

        '''
        action_index = {"UP": 0, "DOWN" : 1, "LEFT" : 2, "RIGHT" : 3}
        if np.random.randint(1) == 0:
            self.Q1[current_state][action_index[action]] = (self.Q1[current_state][action_index[action]] + 
                                               self.alpha * (reward + self.gamma * self.Q2[next_state][np.argmax(self.Q1[next_state])] - self.Q1[current_state][action_index[action]])) 
        else:
            self.Q2[current_state][action_index[action]] = (self.Q2[current_state][action_index[action]] + 
                                               self.alpha * (reward + self.gamma * self.Q1[next_state][np.argmax(self.Q2[next_state])] - self.Q2[current_state][action_index[action]]))
        self.Q_table[current_state][action_index[action]] = self.Q1[current_state][action_index[action]] +  self.Q2[current_state][action_index[action]]

    def get_max_Q_function(self):
        max_Q_table = np.zeros((self.env_row_max, self.env_col_max))
        '''

        your codes here

        '''
        Q_table= self.get_Q_table()
        for i in range(0, self.env_row_max):
            for j in range(0, self.env_col_max):
                max_Q_table[i][j] = np.max(Q_table[(i, j)])

        return max_Q_table
