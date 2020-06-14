from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
import gym
import numpy as np
from gym import wrappers
import tensorflow as tf

LR = 0.01
LAMBDA = 0.7
GAMMA = 0.999

class Model:
    def __init__(self, env, batch, n_components=500):
        self.actions = [0, 1, 2]
        # self.scaler = StandardScaler().fit(batch)
        # self.features_extractor = FeatureUnion([
        #     ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
        #     ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
        #     ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
        #     ("rbf4", RBFSampler(gamma=0.1, n_components=1000))
        #     ])
        # self.features_extractor.fit(self.scaler.transform(batch))
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
                ])
        self.features_extractor = featurizer.fit(self.scaler.transform(observation_examples))

        D = len(self.features_extractor.transform(self.scaler.transform(batch))[0])
        self.w = np.array([np.random.randn(D)/np.sqrt(D) for a in self.actions])
        self.e = np.zeros((len(self.actions), D))

    def transform(self, state):
        features = self.features_extractor.transform(self.scaler.transform([state]))[0]
        return features

    def predict(self, state):
        features = self.transform(state)
        return np.dot(features, self.w.T)

    def partial_fit(self, state, action, G):
        features = self.transform(state)
        pred = self.predict(state)
        delta = G-pred[action]
        self.e = LAMBDA * GAMMA * self.e
        self.e[action] = self.e[action] + features
        self.w[action] += LR*delta*self.e[action]

def play_episode(models, env, eps):
    observation = env.reset()
    done = False
    nb_steps = 0
    states_actions = []
    rewards = []
    while not done and nb_steps<10000:
        nb_steps +=1
        action = np.argmax(models.predict(observation))

        if np.random.rand() < eps:
            action = np.random.choice([0, 1, 2])

        new_observation, reward, done, info = env.step(action)
        #reward += 100 * (abs(new_observation[1]) - abs(observation[1]))
        # if done:
        #     reward += 300

        G = reward + np.max(models.predict(new_observation))
            #print(G)
        models.partial_fit(observation, action, G)
        observation = new_observation
    return nb_steps

env = gym.make('MountainCar-v0').env

def batch():
    batch = []
    for i in range(200):
        done = False
        nb_steps = 0
        env.reset()
        while not done and nb_steps < 200:
            nb_steps +=1
            action = np.random.choice([0, 1, 2])
            observation, reward, done, info = env.step(action)
            batch.append(observation)
    batch = np.array(batch)
    return batch



actions = [0, 1, 2]
models = Model(env, batch())

from tqdm import tqdm
lengths = []
for i in range(200):
    eps = 0.1*(0.97**i)
    nb_steps = play_episode(models, env, eps)
    print(i, nb_steps)
    lengths.append(nb_steps)

import matplotlib.pyplot as plt
plt.plot(lengths)
plt.show()

env = wrappers.Monitor(env, 'videos', force=True)
print(play_episode(models, env, eps=0))
