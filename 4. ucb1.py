import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm

BANDITS_PROBOBALITIES = [0.2, 0.5, 0.7]
NB_RUNS = 10000

class Bandit():
    def __init__(self, p):
        self.p = p
        self.N = 0
        self.estimate_p = 0
        self.pull()

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
    for N in range(1, NB_RUNS+1):
        j = np.argmax([bandit.estimate_p + np.sqrt(2*np.log(N)/bandit.N) for bandit in bandits])
        rewards.append(bandits[j].pull())
        print([bandit.N for bandit in bandits])
    #print('estimates:', [b.estimate_p for b in bandits])
    win_rates = np.cumsum(rewards)/(np.arange(NB_RUNS)+1)
    plt.plot(win_rates)
    plt.plot(NB_RUNS * [np.max(BANDITS_PROBOBALITIES)])
    plt.show()
