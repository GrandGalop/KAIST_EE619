from env1 import Robot_Gridworld
import matplotlib.pyplot as plt
import pdb
from Reinforce import Reinforce
import numpy as np
import torch

gamma = 0.99
returns = []

def update():
    global returns
    step = 0

    for episode in range(1000):

        state = env.reset()

        step_count = 0
        return_value = 0
        Reinforce.saved_rewards = []
        Reinforce.saved_log_probs = []

        while True:

            env.render()

            action, probability = Reinforce.choose_action(state)
            next_state, reward, terminal = env.step(action)

            return_value = reward + (gamma * return_value)
            step_count += 1
            #dqn.store_transition(state, action, reward, next_state)
            Reinforce.saved_rewards.append(reward)
            #### Begin learning after accumulating certain amount of memory #####
            state = next_state
            Reinforce.saved_log_probs.append(torch.log(probability))
            if terminal:

                print(" {} End. Total steps : {}\n".format(episode + 1, step_count))
                break

            if step_count > 1000:
                break

            step += 1
        returns.append(return_value)
        Reinforce.learn() #Reinforce.saved_rewards, Reinforce.saved_log_probs)


    ####### To Do ########
    # Plot average returns per episode
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



    Reinforce = Reinforce(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        discount_factor=0.9,
                        eps=0.1)


    env.after(100, update) #Basic module in tkinter
    env.mainloop() #Basic module in tkinter


