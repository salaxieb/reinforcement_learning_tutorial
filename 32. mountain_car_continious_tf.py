import gym
import numpy as np

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from gym import wrappers

ALPHA = 0.01
GAMMA = 0.999
LR = 0.1


class HiddenLayer:
    def __init__(self, input_size, hidden_size, activation='elu'):
        self.W = tf.Variable(initial_value=tf.random.normal(shape=(input_size, hidden_size)))
        self.b = tf.Variable(initial_value=tf.random.normal(shape=(hidden_size,)))
        self.activation = activation

    def forward(self, X):
        X = tf.matmul(X, self.W) + self.b
        if self.activation == 'elu':
            return tf.nn.elu(X)
        if self.activation == 'softmax':
            return tf.nn.softmax(X)
        return X


class PolicyModel:
    def __init__(self, action_space, observation_space, hidden_layers, batch):
        self.scaler = StandardScaler().fit(batch)

        self.K = observation_space.shape[0]
        self.M = 1

        inp = tf.keras.layers.Input(shape=self.K, name='state')
        X = inp
        X = tf.keras.layers.Dense(10, activation='elu')(X)
        X = tf.keras.layers.Dense(10, activation='elu')(X)
        X = tf.keras.layers.Dense(10, activation='elu')(X)
        mean = tf.keras.layers.Dense(1, activation='tanh',)(X)


        self.mean = tf.keras.models.Model(inputs=inp, outputs=mean)
        self.mean.compile(optimizer='sgd', loss='mse')

        # variance = tf.keras.layers.Dense(1, activation='linear')(X)
        # variance = tf.math.softplus(variance)
        #
        # self.variance = tf.keras.models.Model(inputs=inp, outputs=variance)
        # self.variance.compile(optimizer='sgd', loss='mse')

        self.mean_optimizer=tf.keras.optimizers.Adagrad(0.1)
        # self.variance_optimizer=tf.keras.optimizers.Adagrad(0.1)


        #self.opt = tf.keras.optimizers.Adagrad(0.1)

        # self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)
        # self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-4, momentum=0.9).minimize(cost)
        # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)


    def partial_fit(self, X, actions, advantages):
        X = self.scaler.transform([X])
        # X = self.model.predict(X)
        # print(np.shape(tf.math.log(X)))
        # print(np.shape(advantages))
        # print(np.shape(advantages * tf.math.log(X)))
        # Compute the gradients for a list of variables.
        with tf.GradientTape() as tape:
            mean = self.mean(X)[0]
            loss_mean = tf.math.reduce_sum(tf.math.negative(advantages * tf.math.log(mean)))

            # print('loss', loss_mean)
            vars = self.mean.trainable_weights
            # print('vars', vars)
            grads = tape.gradient(loss_mean, vars)
            # print('grads', grads)
            self.mean_optimizer.apply_gradients(zip(grads, vars))


        # with tf.GradientTape() as tape:
            # print(np.shape(X))
            # variance = self.variance(X)
            # loss_variance = tf.math.reduce_sum(tf.math.negative(advantages * tf.math.log(variance[0])))
            #
            # # print('loss', loss_variance)
            # vars = self.variance.trainable_weights
            # # print('vars', vars)
            # grads = tape.gradient(loss_variance, vars)
            # # print('grads', grads)
            # self.variance_optimizer.apply_gradients(zip(grads, vars))


    def predict(self, X):
        X = self.scaler.transform([X])
        # return self.mean.predict([X])[0], self.variance.predict([X])[0]
        return self.mean.predict([X])[0]

    def sample_action(self, X):
        # mu, var = self.predict(X)
        mu = self.predict(X)
        # if np.random.rand() < 0.001:
        #     print('var', var)
        # return np.random.normal(mu, var)
        return mu


class ValueFunctionEstimator:
    def __init__(self, batch, observation_space, hidden_layers):
        self.scaler = StandardScaler().fit(batch)
        self.K = observation_space.shape[0]
        self.M = 1

        inp = tf.keras.layers.Input(shape=self.K)
        X=inp
        X = tf.keras.layers.Dense(10, activation='elu')(X)
        X = tf.keras.layers.Dense(1, activation='linear')(X)

        self.model = tf.keras.models.Model(inputs=inp, outputs=X)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(0.0001), loss='mse')

        # self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-2, momentum=0.9).minimize(cost)
        # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

    def partial_fit(self, X, Y):
        X = self.scaler.transform([X])
        self.model.train_on_batch(X, [Y])

    def predict(self, X):
        # print(X)
        # print(np.shape(X))
        X = self.scaler.transform([X])
        # print(X)
        # print(np.shape(X))
        return self.model.predict(X)[0]
        #return self.session.run(self.predict_op, feed_dict={self.X: X})[0]


def play_episode(policy_model, value_f_model, env, p=0.8):
    state = env.reset()
    done = False
    nb_steps = 0
    actions = []
    states = []
    advantages = []
    returns = []
    rewards = 0
    reward = 0
    actual_reward = 0
    while not done and nb_steps <= 200:
        nb_steps +=1
        action = policy_model.sample_action(state)
        # print('action', action)
        new_state, reward, done, info = env.step(action)

        actual_reward += reward
        reward += 500 * (abs(new_state[1]) - abs(state[1]))

        # reward -= 1

        if nb_steps == 200:
            reward += -100
        # update the models
        # print(new_state)
        V_next = value_f_model.predict(new_state)[0]
        G = reward + GAMMA*V_next
        advantage = G - value_f_model.predict(state)
        policy_model.partial_fit(state, action, advantage)
        value_f_model.partial_fit(state, G)
        state = new_state
        rewards += reward
    return actual_reward


env = gym.make('MountainCarContinuous-v0').env
# print(dir(env.action_space))
# print(help(env.action_space))

batch = []

for i in range(200):
    done = False
    nb_steps = 0
    env.reset()
    while not done and nb_steps < 2000:
        nb_steps +=1
        action = np.random.uniform(-1, 1)
        observation, reward, done, info = env.step([action])
        batch.append(observation)
# print(np.shape(batch))

batch = np.array(batch)
policy_model = PolicyModel(env.action_space, env.observation_space, [], batch)#batch[np.random.randint(0, len(batch), size=10)], batch)
value_function_model = ValueFunctionEstimator(batch, env.observation_space, [10])
# init = tf.global_variables_initializer()
# session = tf.InteractiveSession()
# session.run(init)
# policy_model.set_session(session)
# value_function_model.set_session(session)

lengths = []

from tqdm import tqdm
for i in range(50):
    print(i)
    nb_steps = play_episode(policy_model, value_function_model, env)
    print(nb_steps)
    lengths.append(nb_steps)

import matplotlib.pyplot as plt
plt.plot(lengths)
plt.show()

env = wrappers.Monitor(env, 'videos', force=True)
print(play_episode(policy_model, value_function_model, env))
