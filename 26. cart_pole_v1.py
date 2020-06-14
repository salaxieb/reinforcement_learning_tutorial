import gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

ALPHA = 0.01
GAMMA = 0.9
LR = 0.1

class Model:
    def __init__(self, samples, batch):
        self.scaler = StandardScaler().fit(batch)
        self.features_extractor = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000))
            ])
        self.features_extractor.fit(self.scaler.transform(samples))

        self.W = [np.random.normal(size=len(self.features_extractor.transform(samples)[0])), np.random.normal(size=len(self.features_extractor.transform(samples)[0]))]

    def transform(self, observation):
        features = self.features_extractor.transform(self.scaler.transform([observation]))[0]
        return features

    def predict(self, state):
        features = self.transform(state)
        Q = [np.dot(w, features) for w in self.W]
        return Q

    def partial_fit(self, state, reward, action, next_state):
        features = self.transform(state)
        q = (reward + GAMMA*np.max(self.predict(next_state)) - self.predict(state)[action])
        self.W[action] = self.W[action] + LR*q*features


def play_episode(model, env, p=0.8):
    state = env.reset()
    done = False
    nb_steps = 0
    while not done and nb_steps < 2000:
        nb_steps +=1
        action = np.argmax(model.predict(state))

        if np.random.rand() > p:
            action = 1 - action #choosing another action

        new_state, reward, done, info = env.step(action)
        if done:
            reward = -100

        model.partial_fit(state, reward, action, new_state)
        state = new_state
    return nb_steps


env = gym.make('CartPole-v0').env
# print(dir(env.action_space))
# print(help(env.action_space))

batch = []

for i in range(200):
    done = False
    nb_steps = 0
    env.reset()
    while not done and nb_steps < 2000:
        nb_steps +=1
        action = np.random.choice([0, 1])
        observation, reward, done, info = env.step(action)
        batch.append(observation)

batch = np.array(batch)
model = Model(batch, batch)#batch[np.random.randint(0, len(batch), size=10)], batch)
lengths = []

from tqdm import tqdm
for i in tqdm(range(10000)):
    nb_steps = play_episode(model, env)
    lengths.append(nb_steps)

import matplotlib.pyplot as plt
plt.plot(lengths)
plt.show()
