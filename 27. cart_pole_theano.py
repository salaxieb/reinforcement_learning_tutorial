import gym
import numpy as np
import theano
import theano.tensor as T

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
        D = len(self.features_extractor.transform(samples)[0])
        self.w = np.random.randn(D)/np.sqrt(D)
        self.w = theano.shared(self.w)
        X = T.matrix('X')
        Y = T.vector('Y')
        pred = X.dot(self.w)
        delta = Y - pred
        cost = delta.dot(delta)
        grad  = T.grad(cost, self.w)
        updates = [(self.w, self.w - LR*grad)]

        self.train_op = theano.function(
            inputs=[X, Y],
            updates=updates,
        )

        self.predict_op = theano.function(
              inputs=[X],
              outputs=pred,
        )

    def transform(self, state):
        features = self.features_extractor.transform(self.scaler.transform([state]))
        return features

    def predict(self, state):
        features = self.transform(state)
        return self.predict_op(features)

    def partial_fit(self, state, reward, next_state):
        features = self.transform(state)
        q = reward + GAMMA*np.max(self.predict(next_state))
        self.train_op(features, [q])


def play_episode(models, env, p=0.8):
    state = env.reset()
    done = False
    nb_steps = 0
    while not done and nb_steps < 2000:
        nb_steps +=1
        action = np.argmax([model.predict(state) for model in models])

        if np.random.rand() > p:
            action = 1 - action #choosing another action

        new_state, reward, done, info = env.step(action)
        if done:
            reward = -100

        models[action].partial_fit(state, reward, new_state)
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
models = [Model(batch, batch), Model(batch, batch)]#batch[np.random.randint(0, len(batch), size=10)], batch)
lengths = []

from tqdm import tqdm
for i in tqdm(range(10000)):
    nb_steps = play_episode(models, env)
    print(nb_steps)
    lengths.append(nb_steps)

import matplotlib.pyplot as plt
plt.plot(lengths)
plt.show()
