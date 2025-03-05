# **Readme - 강화학습이론 KAIST_EE619**
## **Overview**
Project 1: Implementing tabular Q-learning, Double Q-learning algorithms

Project 2: Implementing Deep Q-Network (DQN), REINFORCE algorithms

Project 3: Implementing the algorithm that you want to implement: Q Actor-Critic

## **Project 1**
- Implement and compare Q-learning & Double Q-learning algorithms.
- Train an agent in a GridWorld environment using reinforcement learning.
- Analyze the learning behavior by visualizing policies and evaluating Q-values.

### **Q-learning**
- A model-free reinforcement learning algorithm that learns an optimal action-selection policy for an agent.
- Uses a **Q-table** that stores Q-values for each **state-action pair**.
- Implements **ε-greedy policy** to balance exploration and exploitation.
- Updates Q-values using the **Bellman equation**:

  $$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]$$

  where:
  - $\( s \)$ = current state
  - $\( a \)$ = action taken
  - $\( s' \)$ = next state
  - $\( \alpha \)$ = learning rate
  - $\( \gamma \)$ = discount factor
  - $\( r \)$ = reward received

### **Double Q-learning**
- A variant of Q-learning that reduces **overestimation bias** in value estimation.
- Maintains **two Q-tables**: \( Q_1 \) and \( Q_2 \).
- Updates one of the two Q-values randomly at each step:
  - If updating **\( Q_1 \)**:

    $$Q_1(s,a) \leftarrow Q_1(s,a) + \alpha \left[ r + \gamma Q_2(s', \arg\max Q_1(s', a')) - Q_1(s,a) \right]$$

  - If updating **\( Q_2 \)**:

    $$Q_2(s,a) \leftarrow Q_2(s,a) + \alpha \left[ r + \gamma Q_1(s', \arg\max Q_2(s', a')) - Q_2(s,a) \right]$$

- **Final Q-value** is computed as:

  $$Q(s, a) = Q_1(s, a) + Q_2(s, a)$$

- **Key Advantage:**  
  - Prevents **overestimation of Q-values** by decoupling the action selection and action evaluation steps.
 
## **Project 2**
- Implement and compare Deep Q-learning & REINFORCE algorithms.
- Train an agent in a GridWorld environment using reinforcement learning.
- Analyze the learning behavior by visualizing policies and evaluating Q-values.

### **Deep Q-Network (DQN)**
- A **value-based** reinforcement learning algorithm.
- Uses a **deep neural network** to approximate Q-values instead of a Q-table.
- Implements **experience replay** and **target networks** to improve stability.

#### **Key Components of DQN**
1. **Experience Replay**  
   - Stores past experiences **(state, action, reward, next state)** in a replay buffer.  
   - Samples **mini-batches** for training to reduce correlation between updates.

2. **Target Network**  
   - Maintains a separate, periodically updated **target Q-network**.  
   - Prevents rapid changes in Q-values, improving training stability.

3. **Q-Value Approximation**  
   - Uses a deep neural network to estimate Q-values:  
   
     $$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q(s,a) \right]$$

### **REINFORCE (Policy Gradient Method)**
- A **policy-based** reinforcement learning algorithm.
- Directly learns a **stochastic policy** instead of estimating Q-values.
- Uses the **log probability of actions** to adjust weights based on rewards.

#### **Key Components of REINFORCE**
1. **Policy Network**  
   - A neural network outputs **action probabilities** instead of Q-values.
   - Selects actions using **softmax probability distribution**.

2. **Policy Gradient Update**  
   - Updates policy parameters using gradient ascent:  
   
     $$\theta \leftarrow \theta + \alpha \sum_{t} G_t \nabla \log \pi_{\theta} (a_t | s_t)$$

   - Where \( G_t \) is the **discounted return** from timestep \( t \).
   - Encourages actions that led to high rewards.

3. **Monte Carlo Updates**  
   - Uses **full episode rewards** to update weights.
   - High variance but unbiased updates.

## **Project3**
Implementing the **Q Actor-Critic (QAC)** algorithm, which combines value-based Q-learning with policy-based Actor-Critic reinforcement learning. The goal is to efficiently train an agent using **both policy gradients and Q-value estimates**.
The **Q Actor-Critic (QAC)** algorithm evaluates actions using a **Q-function**, representing the expected return when the agent takes action \( a \) in state \( s \). The policy network updates its parameters via **policy gradients**, while the critic network updates Q-values using **TD(0) learning**.

### **1. Q-Value Estimation (Critic)**
- The **critic network** estimates the Q-value:  
  $$Q(s, a) \approx \mathbb{E}[R | s, a]$$
- Updated using **Temporal Difference (TD(0)) Learning**:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### **2. Policy Update (Actor)**
- The **policy network** selects actions based on observations.
- The actor’s gradient update is computed using the critic’s Q-value:
  $$\nabla_{\theta} J(\theta) = \mathbb{E} \left[ \nabla_{\theta} \log \pi_{\theta} (a | s) \cdot Q(s, a) \right]$$
- Encourages **high Q-value actions** by maximizing the expected return.

### **Implementation Details**
#### **1. Policy & Critic Networks**
- The model consists of **two networks**:
  - **Policy Network**: Outputs an action distribution.
  - **Critic Network**: Estimates Q-values.

- **Structure**:
  - **Input**: 24-dimensional observation space.
  - **Output**: 6-dimensional action space.
  - **Hidden layers**: 3-layer MLP (Multi-Layer Perceptron).

#### **2. Training Process**
- **Step 1**: The policy network takes an **observation (state)** and outputs an **action**.
- **Step 2**: The action is executed, and the environment returns a **reward**.
- **Step 3**: The critic network updates **Q-values** using **TD(0)** learning.
- **Step 4**: The policy network updates based on **actor-critic gradients**.
