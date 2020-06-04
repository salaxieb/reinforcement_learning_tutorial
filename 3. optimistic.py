import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm

BANDITS_PROBOBALITIES = [2, 3, 4]
NB_RUNS = 10000

class Bandit():
    def __init__(self, p):
        self.p = p
        self.N = 1
        self.estimate_p = 10

    def pull(self):
        reward = np.random.normal() + self.p
        self.update(reward)
        return reward

    def update(self, reward):
        self.N += 1
        self.estimate_p = 1/self.N*reward + (self.N-1)/self.N * self.estimate_p

if __name__ == '__main__':
    bandits = [Bandit(p) for p in BANDITS_PROBOBALITIES]
    nb_optimal_choice = 0
    nb_non_optimal_choice = 0
    rewards = []
    for i in range(NB_RUNS):
        j = np.argmax([bandit.estimate_p for bandit in bandits])
        rewards.append(bandits[j].pull())
        print([bandit.estimate_p for bandit in bandits])
    #print('estimates:', [b.estimate_p for b in bandits])
    win_rates = np.cumsum(rewards)/(np.arange(NB_RUNS)+1)
    plt.plot(win_rates)
    plt.plot(NB_RUNS * [np.max(BANDITS_PROBOBALITIES)])
    plt.show()
