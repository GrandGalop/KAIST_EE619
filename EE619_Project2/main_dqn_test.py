from env2 import Robot_Gridworld
from DQN_test import DeepQLearning
import matplotlib.pyplot as plt
import pdb
import numpy as np


gamma = 0.99
step = 0

def update():
    global step
    returns = []
    for episode in range(1000):

        state = env.reset()

        step_count = 0
        return_value = 0

        while True:

            env.render() # different with self.update

            action = dqn.choose_action(state)

            next_state, reward, terminal = env.step(action)

            return_value = reward + (gamma * return_value)
            step_count += 1
            dqn.store_transition(state, action, reward, next_state, terminal)

            if (step > 200) and (step % 5 == 0):
                dqn.learn()
            #### Begin learning after accumulating certain amount of memory #####
            state = next_state

            if terminal:

                print(" {} End. Total steps : {}\n".format(episode + 1, step_count))
                break

            step += 1
    ####### To Do ########
    # Plot average returns per episode
        returns.append(return_value)
    plt.figure
    plt.plot(range(1000), returns)
    plt.xlabel("# of Episodes")
    plt.ylabel("Return")
    plt.title("Return per Episode")
    plt.show()
    returns = []

    print('Game over.\n')
    env.destroy()


if __name__ == "__main__":

    env = Robot_Gridworld()

    dqn = DeepQLearning(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        discount_factor=0.9,
                        e_greedy=0.05,
                        replace_target_iter=50,
                        memory_size=3000,
                        batch_size=32)


    env.after(100, update) #Basic module in tkinter
    env.mainloop() #Basic module in tkinter


