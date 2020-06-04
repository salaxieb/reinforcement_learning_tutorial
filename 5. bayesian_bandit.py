import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
# from tqdm import tqdm

BANDITS_PROBOBALITIES = [0.2, 0.5, 0.7]
NB_RUNS = 10000
BREAKPOINTS = [1, 5, 10, 100, 1000, 9999]

class Bandit():
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1

    def pull(self):
        if np.random.rand() < self.p:
            reward = 1
            self.a += 1
        else:
            reward = 0
            self.b += 1
        return reward

    def beta(self):
        y = np.arange(0, 1, 0.01)
        rv = beta.pdf(y, self.a, self.b)
        return rv

    def sample(self):
        return np.random.beta(self.a, self.b)

if __name__ == '__main__':
    bandits = [Bandit(p) for p in BANDITS_PROBOBALITIES]
    nb_optimal_choice = 0
    nb_non_optimal_choice = 0
    rewards = []
    for N in range(1, NB_RUNS+1):
        j = np.argmax([bandit.sample() for bandit in bandits])
        rewards.append(bandits[j].pull())
        if N in BREAKPOINTS:
            print([(bandit.a, bandit.b) for bandit in bandits])
            for bandit in bandits:
                plt.plot(bandit.beta())
            plt.show()

    #print('estimates:', [b.estimate_p for b in bandits])
    win_rates = np.cumsum(rewards)/(np.arange(NB_RUNS)+1)
    plt.plot(win_rates)
    plt.plot(NB_RUNS * [np.max(BANDITS_PROBOBALITIES)])
    plt.show()
