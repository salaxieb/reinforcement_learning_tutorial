from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import gym
import numpy as np
from gym import wrappers
import tensorflow as tf

ALPHA = 0.1
GAMMA = 0.9
N = 10

class Model:
    def __init__(self, batch):
        self.actions = [0, 1, 2]
        self.scaler = StandardScaler().fit(batch)
        D = len(self.scaler.transform(batch)[0])
        rate = 0.2
        inp = tf.keras.layers.Input(shape=D)
        X = inp
        X = tf.keras.layers.Dense(10, activation='elu')(X)
        X = tf.keras.layers.Dropout(rate)(X)
        X = tf.keras.layers.Dense(7, activation='elu')(X)
        X = tf.keras.layers.Dropout(rate)(X)
        X = tf.keras.layers.Dense(len(self.actions), activation='linear')(X)
        self.m = tf.keras.models.Model(inputs=inp, outputs=X)
        self.m.compile(optimizer='sgd', loss='mse')

    def transform(self, state):
        features = self.scaler.transform([state])
        return features

    def predict(self, state):
        features = self.transform(state)
        return self.m.predict(features)[0]

    def partial_fit(self, state, action, G):
        features = self.transform(state)
        pred = self.predict(state)
        Y = [pred[i] if i!= action else G for i in range(len(self.actions))]
        self.m.train_on_batch(features, np.array([Y]))

def play_episode(model, env):
    observation = env.reset()
    done = False
    nb_steps = 0
    p = 0.9
    states_actions = []
    rewards = []
    while not done and nb_steps<400:
        nb_steps +=1
        action = np.argmax(model.predict(observation))
        # if np.random.rand() > p:
        #     action = np.random.choice([0, 1, 2])

        new_observation, reward, done, info = env.step(action)
        reward += 300 * (abs(new_observation[1]) - abs(observation[1]))
        if done:
            reward += 300

        states_actions.append((observation, action))
        rewards.append(reward)

        if len(rewards) == N:
            state, action = states_actions.pop(0)
            return_ = 0
            for i, r in enumerate(rewards):
                return_ += GAMMA**i * r

            rewards.pop(0)
            G = return_ + np.max(model.predict(new_observation))
            #print(G)
            model.partial_fit(state, action, G)
        observation = new_observation

    #print(rewards)

    while rewards:
        state, action = states_actions.pop(0)
        return_ = 0
        for i, r in enumerate(rewards):
            return_ += GAMMA**i * r

        rewards.pop(0)
        G = return_ + np.max(model.predict(new_observation))

        #print(G)
        #print(rewards)
        model.partial_fit(state, action, G)
    return nb_steps

env = gym.make('MountainCar-v0').env

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
model = Model(batch)#batch[np.random.randint(0, len(batch), size=10)], batch)
lengths = []

model = Model(batch)

from tqdm import tqdm
for i in tqdm(range(10000)):
    nb_steps = play_episode(model, env)
    print(nb_steps)
    lengths.append(nb_steps)

import matplotlib.pyplot as plt
plt.plot(rewards)
plt.show()

env = wrappers.Monitor(env, 'videos', force=True)
print(play_episode(model, env))
