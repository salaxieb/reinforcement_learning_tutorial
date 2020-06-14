import gym
import numpy as np

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from gym import wrappers

ALPHA = 0.01
GAMMA = 0.9
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
        self.M = action_space.n

        inp = tf.keras.layers.Input(shape=self.K, name='state')
        X = inp
        probs = tf.keras.layers.Dense(self.M, activation='softmax')(X)

        advantage = tf.keras.Input(shape=1, name='advantage')

        self.optimizer=tf.keras.optimizers.Adagrad(0.1)

        self.model = tf.keras.models.Model(inputs=inp, outputs=probs)
        self.model.compile(optimizer='sgd', loss='mse')

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
            X = self.model(X)
            # for layer in self.model.layers:
            #     X = layer(X)
            # loss = tf.math.reduce_mean(advantages * tf.math.log(X))
            loss = tf.math.reduce_sum(tf.math.negative(advantages * tf.math.log(X[0])))

        # print('loss', loss)
        vars = self.model.trainable_weights
        # print('vars', vars)
        grads = tape.gradient(loss, vars)
        # print('grads', grads)

        # Process the gradients, for example cap them, etc.
        # capped_grads = [MyCapper(g) for g in grads]
        #processed_grads = [process_gradient(g) for g in grads]

        # Ask the optimizer to apply the processed gradients.
        self.optimizer.apply_gradients(zip(grads, vars))
        # loss = tf.math.reduce_sum(np.negative(advantages * tf.math.log(X)))
        # self.optimizer.minimize(loss)

        # self.model.train_on_batch


    def predict(self, X):
        X = self.scaler.transform([X])
        return self.model.predict([X])[0]
        # X = self.session.run(self.predict_op, feed_dict={self.X: X})[0]
        # return X

    def sample_action(self, X):
        p = self.predict(X)
        return np.random.choice(len(p), p=p)


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
        #self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

    def partial_fit(self, X, Y):
        X = self.scaler.transform([X])
        self.model.train_on_batch(X, [Y])

    def predict(self, X):
        X = self.scaler.transform([X])
        return self.model.predict([X])[0]
        #return self.session.run(self.predict_op, feed_dict={self.X: X})[0]


def play_episode(policy_model, value_f_model, env, p=0.8):
    state = env.reset()
    done = False
    nb_steps = 0
    actions = []
    states = []
    advantages = []
    returns = []
    rewards = []
    reward = 0
    while not done and nb_steps < 500:
        nb_steps +=1
        action = policy_model.sample_action(state)

        # states.append(state)
        # actions.append(action)
        # rewards.append(reward)

        new_state, reward, done, info = env.step(action)

        if done:
            reward = -100


        # update the models
        V_next = value_f_model.predict(new_state)[0]
        G = reward + GAMMA*V_next
        advantage = G - value_f_model.predict(state)
        policy_model.partial_fit(state, action, advantage)
        value_f_model.partial_fit(state, G)

        #return_ = reward + GAMMA * value_f_model.predict(new_state)
        state = new_state

    # # save the final (s,a,r) tuple
    # action = policy_model.sample_action(state)
    # states.append(state)
    # actions.append(action)
    # rewards.append(reward)
    #
    # returns = []
    # advantages = []
    # G = 0
    # for s, r in zip(reversed(states), reversed(rewards)):
    #     returns.append(G)
    #     advantages.append(G - value_f_model.predict(s))
    #     G = r + GAMMA*G
    # returns.reverse()
    # advantages.reverse()
    #
    # # update the models
    # policy_model.partial_fit(states, actions, advantages)
    # value_f_model.partial_fit(states, returns)

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
policy_model = PolicyModel(env.action_space, env.observation_space, [], batch)#batch[np.random.randint(0, len(batch), size=10)], batch)
value_function_model = ValueFunctionEstimator(batch, env.observation_space, [10])
# init = tf.global_variables_initializer()
# session = tf.InteractiveSession()
# session.run(init)
# policy_model.set_session(session)
# value_function_model.set_session(session)

lengths = []

from tqdm import tqdm
for i in range(100):
    nb_steps = play_episode(policy_model, value_function_model, env)
    print(nb_steps)
    lengths.append(nb_steps)

import matplotlib.pyplot as plt
plt.plot(lengths)
plt.show()

env = wrappers.Monitor(env, 'videos', force=True)
print(play_episode(policy_model, value_function_model, env))
