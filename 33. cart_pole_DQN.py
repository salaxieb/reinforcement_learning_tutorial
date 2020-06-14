import gym
import numpy as np

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from gym import wrappers

ALPHA = 0.01
GAMMA = 0.999
LR = 0.1


class ActionValueFunctionEstimator:
    def __init__(self, batch, action_space):
        self.scaler = StandardScaler().fit(batch)
        def untransform(vec):
            return np.array(vec)

        self.scaler.transform = untransform
        self.len_inputs = len(self.scaler.transform(batch)[0])

        print(self.len_inputs)

        self.len_actions = action_space.n

        inp = tf.keras.layers.Input(shape=self.len_inputs)
        X=inp
        X = tf.keras.layers.Dense(200, activation='elu')(X)
        X = tf.keras.layers.Dense(200, activation='elu')(X)

        turn_left = tf.keras.layers.Dense(100, activation='elu')(X)
        turn_right = tf.keras.layers.Dense(100, activation='elu')(X)

        turn_left = tf.keras.layers.Dense(1, activation='linear')(turn_left)
        turn_right = tf.keras.layers.Dense(1, activation='linear')(turn_right)

        X = tf.keras.layers.Concatenate()([turn_left, turn_right])

        self.model = tf.keras.models.Model(inputs=inp, outputs=X)
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=0.001), loss='mse')
        # tf.keras.optimizers.Adagrad(lr=0.01, decay=0.1)

        self.states = []
        self.actions = []
        self.new_states = []
        self.rewards = []
        self.dones = []
        self.batch_size = 32

    def partial_fit(self, state, action, action_value):
        X = self.scaler.transform([state])

        Y = [0] * self.len_actions
        Y[action] = action_value
        Y[1-action] = self.predict(state)[1-action]
        self.model.train_on_batch(X, np.array([Y]))

    def train(self, target_network):
        if len(self.states) < 1000:
            return
        # while len(self.states) > 2 * self.batch_size:
            # idx = list(reversed(range(len(self.states))))[:self.batch_size]
        idx = np.random.choice(range(len(self.states)), self.batch_size)
        batch_x = [self.states[i] for i in idx]
        batch_x = self.scaler.transform(batch_x)

        batch_y = []
        for i in idx:
            if not self.dones[i]:
                return_ = self.rewards[i] + np.max(target_network.predict(self.new_states[i]))
            else:
                return_ = self.rewards[i]
            y = [0, 0]
            y[self.actions[i]] = return_
            y[1 - self.actions[i]] = target_network.predict(self.states[i])[1 - self.actions[i]]
            batch_y.append(y)

        # idx = list(reversed(sorted(list(set(idx)))))
        # for i in idx:
        #     self.states.pop(i)
        #     self.actions.pop(i)
        #     self.new_states.pop(i)
        #     self.rewards.pop(i)
        #     self.dones.pop(i)
        self.model.train_on_batch(np.array(batch_x), np.array(batch_y))

    def predict(self, state):
        X = self.scaler.transform([state])
        return self.model.predict(X)[0]
        #return self.session.run(self.predict_op, feed_dict={self.X: X})[0]

    def sample_action(self, state, p=0.1):
        a_v_e = self.predict(state)
        # print(a_v_e)
        optimal = np.argmax(a_v_e)
        if np.random.rand() < p:
            return 1-optimal
        return optimal

    def add_sars(self, state, action, reward, new_state, done):
        if len(self.states) > 10000:
            self.states.pop(0)
            self.actions.pop(0)
            self.new_states.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
        self.states.append(state)
        self.actions.append(action)
        self.new_states.append(new_state)
        self.rewards.append(reward)
        self.dones.append(done)

class TargetNetwork(ActionValueFunctionEstimator):
    def __init__(self, model, scaler):
        self.model = tf.keras.models.clone_model(model)
        self.scaler = scaler


def play_episode(action_value_function_model, env, target_network, p=0.1):
    observation = env.reset()
    observations = [observation] * FRAMES
    state = np.reshape(observations[-FRAMES:], FRAMES*4)

    done = False
    nb_steps = 0
    sars = []

    # target_network = TargetNetwork(action_value_function_model.model, action_value_function_model.scaler)

    while not done and nb_steps < 500:
        nb_steps +=1
        action = action_value_function_model.sample_action(state, p)
        # if nb_steps%100 == 0:
            # target_network = TargetNetwork(action_value_function_model.model, action_value_function_model.scaler)
            # for state, action, reward, new_state in reversed(sars):
            #     G = reward + GAMMA * np.max(action_value_function_model.predict(new_state))
            #     action_value_function_model.partial_fit(state, action, G)
            # sars = []

        new_observation, reward, done, info = env.step(action)

        observations.append(new_observation)
        new_state = np.reshape(observations[-FRAMES:], FRAMES*4)

        if done:
            reward = -200

        action_value_function_model.add_sars(state, action, reward, new_state, done)

        state = new_state

    action_value_function_model.train(target_network)

    return nb_steps


env = gym.make('CartPole-v0').env
# print(dir(env.action_space))
# print(help(env.action_space))

batch = []

FRAMES = 1
for i in range(20):
    done = False
    nb_steps = 0
    observation = env.reset()
    observations = [observation] * FRAMES
    while not done and nb_steps < 20:
        nb_steps +=1
        action = np.random.choice([0, 1])
        observation, reward, done, info = env.step(action)
        observations.append(observation)
        state = np.reshape(observations[-FRAMES:], FRAMES*4)
        batch.append(state)

batch = np.array(batch)
# print(np.shape(batch))
action_value_function_model = ActionValueFunctionEstimator(batch, env.action_space)


lengths = []

from tqdm import tqdm
running_avg = []
for i in range(1000):
    if i%50==0:
        target_network = TargetNetwork(action_value_function_model.model, action_value_function_model.scaler)
    p = 0.5*0.99**i
    nb_steps = play_episode(action_value_function_model, env, target_network, p)
    lengths.append(nb_steps)
    avg = np.mean(lengths[-min(100, len(lengths)):])
    print(i, avg, action_value_function_model.predict([0, 0, 0, 0]))
    running_avg.append(avg)

import matplotlib.pyplot as plt
plt.plot(running_avg)
plt.show()

env = wrappers.Monitor(env, 'videos', force=True)
print(play_episode(action_value_function_model, env, target_network, p=0))
