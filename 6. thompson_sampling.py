import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# from tqdm import tqdm

BANDITS_PROBOBALITIES = [0.2, 0.5, 0.7]
NB_RUNS = 10000
BREAKPOINTS = [1, 5, 10, 100, 1000, 9999]

class Bandit():
    def __init__(self, p):
        self.p = p
        self.tau = 0.1
        self.N = 0
        self.liambda0 = 1
        self.m0 = 0.5
        self.m = self.liambda0 * self.m0
        self.liambda = self.tau * self.N + self.liambda0

    def pull(self):
        reward = np.random.normal() + self.p
        self.N += 1
        liambda = self.tau * self.N + self.liambda0
        liambda_new = liambda + self.tau
        self.m = self.m * liambda/liambda_new + 1/liambda_new * self.tau * reward
        liambda = liambda_new
        return reward

    def norm(self):
        y = np.arange(0, 1, 0.01)
        rv = norm.pdf(y, loc=self.m, scale=self.tau)
        return rv

    def sample(self):
        return np.random.normal(self.m, self.tau)

if __name__ == '__main__':
    bandits = [Bandit(p) for p in BANDITS_PROBOBALITIES]
    nb_optimal_choice = 0
    nb_non_optimal_choice = 0
    rewards = []
    for N in range(1, NB_RUNS+1):
        j = np.argmax([bandit.sample() for bandit in bandits])
        rewards.append(bandits[j].pull())
        if N in BREAKPOINTS:
            print([bandit.m for bandit in bandits])
            for bandit in bandits:
                plt.plot(bandit.norm(), label=f'bandit with p = {bandit.p} m = {bandit.m} and pulls {bandit.N}')
            plt.legend()
            plt.show()

    #print('estimates:', [b.estimate_p for b in bandits])
    win_rates = np.cumsum(rewards)/(np.arange(NB_RUNS)+1)
    plt.plot(win_rates)
    plt.plot(NB_RUNS * [np.max(BANDITS_PROBOBALITIES)])
    plt.show()
