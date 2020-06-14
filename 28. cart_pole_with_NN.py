import gym
import numpy as np

from sklearn.preprocessing import StandardScaler
import tensorflow as tf

ALPHA = 0.01
GAMMA = 0.9
LR = 0.1

class Model:
    def __init__(self, batch):
        self.scaler = StandardScaler().fit(batch)
        D = len(self.scaler.transform(batch)[0])
        rate = 0.2
        inp = tf.keras.layers.Input(shape=D)
        X = inp
        X = tf.keras.layers.Dense(10, activation='elu')(X)
        X = tf.keras.layers.Dropout(rate)(X)
        X = tf.keras.layers.Dense(7, activation='elu')(X)
        X = tf.keras.layers.Dropout(rate)(X)
        X = tf.keras.layers.Dense(2, activation='linear')(X)
        self.m = tf.keras.models.Model(inputs=inp, outputs=X)
        self.m.compile(optimizer='sgd', loss='mse')

    def transform(self, state):
        features = self.scaler.transform([state])
        return features

    def predict(self, state):
        features = self.transform(state)
        return self.m.predict(features)[0]

    def partial_fit(self, state, action, reward, next_state):
        features = self.transform(state)
        q = reward + GAMMA*np.max(self.predict(next_state))
        Y = [0, 0]
        Y[action] = q
        Y[1-action] = self.predict(state)[1-action]
        self.m.train_on_batch(features, np.array([Y]))


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

        model.partial_fit(state, action, reward, new_state)
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
model = Model(batch)#batch[np.random.randint(0, len(batch), size=10)], batch)
lengths = []

from tqdm import tqdm
for i in tqdm(range(10000)):
    nb_steps = play_episode(model, env)
    print(nb_steps)
    lengths.append(nb_steps)

import matplotlib.pyplot as plt
plt.plot(lengths)
plt.show()
