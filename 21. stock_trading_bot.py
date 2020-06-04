import pandas as pd
import numpy as np
import keras
import math
import keras.backend as K
from sklearn.preprocessing import StandardScaler

prices = pd.read_csv('aapl_msi_sbux.csv', sep=',')
ALPHA = 0.2
GAMMA = 0.99

class Environment:
    def __init__(self, prices):
        self.prices = prices
        self.reset()
        self.trainlimit = len(prices)/2
        self.testlimit = len(prices)-1

    def reset(self):
        self.timestamp = -1

    def current_prices(self):
        return self.prices.iloc[self.timestamp]

    def future_prices(self):
        return self.prices.iloc[self.timestamp+1]

    def act(self, action, amount, cash, share_name):
        if action == 'buy':
            amount, change = self.buy(cash, share_name)
            return amount, change
        if action == 'sell':
            _return_ = self.sell(amount, share_name)
            return 0, _return_
        if action == 'hold':
            amount, change = amount, cash
            return amount, change
        raise Exception

    def sell(self, amount, share_name):
        sell_price = 1*self.current_prices()[share_name]
        return sell_price * amount

    def if_all_sold_tomorrow(self, amount, share_name):
        sell_price = 1*self.current_prices()[share_name]
        return sell_price * amount

    def buy(self, cash, share_name):
        buy_price = 1*self.current_prices()[share_name]
        amount = math.floor(cash/buy_price)
        change = cash - amount * buy_price
        return amount, change

class Agent:
    def __init__(self, env):
        self.reset()
        self.target_length = 20
        self.prices_history = []
        self.env = env
        self.model = self.init_model()
        self.actions_dict = {0: 'buy',
                             1: 'hold',
                             2: 'sell'}

        self.shares_dict = {'AAPL': 0,
                       'MSI':  1,
                       'SBUX': 2}
        self.x_scaler = StandardScaler()
        self.prices_scaler = StandardScaler()
        self.return_scaler = StandardScaler()

    def funds_reset(self):
        self.budget = 20000 #$
        self.initial_budget = self.budget
        self.portfolio = {'AAPL':0,
                          'MSI':0,
                          'SBUX':0}

    def reset(self):
        self.funds_reset()
        self.prices_history = []

    def wait_till_prices_update(self):
        self.env.timestamp += 1
        prices = []
        prices.append(self.env.current_prices())
        prices.append(self.env.current_prices())
        prices.append(self.env.current_prices())
        prices = self.prices_scaler.partial_fit(prices).transform(prices)[0]
        self.prices_history.append(prices)

    @staticmethod
    def normilize(value, mean, sigma):
        return (value - mean)/sigma

    @staticmethod
    def my_loss(X, prediction):
        loss = K.mean(keras.layers.concatenate(
                                                [keras.losses.categorical_crossentropy(X[:, 0:3], prediction[:, 0:3]),
                                                 keras.losses.categorical_crossentropy(X[:, 3:6], prediction[:, 3:6]),
                                                 keras.losses.categorical_crossentropy(X[:, 6: ], prediction[:, 6: ])]
                                                 ))
        return loss

    @staticmethod
    def my_activation(X):
        X = [
             keras.layers.Activation('softmax')(X[:, 0:3]),
             keras.layers.Activation('softmax')(X[:, 3:6]),
             keras.layers.Activation('softmax')(X[:, 7: ])
             ]
        X = keras.layers.Concatenate(axis=-1)(X)
        return X

    def init_model(self):
        agent_state = keras.layers.Input(shape=(4,))
        X = agent_state
        X = keras.layers.Dense(30, activation='elu')(X)
        X = keras.layers.Dense(30, activation='elu')(X)
        a_s = keras.layers.Dense(30, activation='elu')(X)

        market_state = keras.layers.Input(shape=(self.target_length, 3))
        X = market_state
        X = keras.layers.LSTM(10, return_sequences=True)(X)
        #X = keras.layers.LSTM(10, return_sequences=True)(X)
        X = keras.layers.LSTM(10)(X)
        m_s = keras.layers.Dense(30, activation='elu')(X)
        X = keras.layers.Concatenate()([a_s, m_s])

        X = keras.layers.Dense(30, activation='elu')(X)
        #X = keras.layers.Dense(10, activation='elu')(X)
        X = keras.layers.Dense(27, activation='linear')(X)
        #X = keras.layers.Lambda(self.my_activation)(X)
        model = keras.models.Model(inputs=[agent_state, market_state], outputs=X)
        model.compile(optimizer='adam', loss='mse')#self.my_loss)
        model.summary()
        return model

    @staticmethod
    def pad_sequences(arr, target_len):
        if len(arr) < target_len:
            zeros = np.zeros((target_len-len(arr), len(arr[0])))
            return np.concatenate((zeros, np.array(arr)))
        else:
            return arr[-target_len:]

    def features_generator(self, state):
        budget = state[0]
        portfolio = state[1]
        x = []
        x.append(budget)
        x.append(portfolio['AAPL'])
        x.append(portfolio['MSI'])
        x.append(portfolio['SBUX'])
        x = self.x_scaler.partial_fit([x]).transform([x])
        prices_history = self.pad_sequences(self.prices_history, target_len=self.target_length)
        return x, np.array([prices_history])

    def current_state(self):
        return [self.budget, self.portfolio.copy()]

    def set_state(self, new_state):
        self.budget = new_state[0]
        self.portfolio = new_state[1].copy()

    def _get_reward(self, budget, portfolio):
        cash = budget
        portfolio = portfolio.copy()
        for share_name in self.shares_dict:
            _return_ = env.if_all_sold_tomorrow(portfolio[share_name], share_name)
            portfolio[share_name] = 0
            budget += _return_
        return budget - cash

    def get_new_state(self, state, action):
        budget = state[0]
        portfolio = state[1].copy()
        #first sell then buy (to have extra money on hands)
        for share_name in self.shares_dict:
            a = action[self.shares_dict[share_name]]
            a_string = self.actions_dict[a]
            if a_string=='sell':
                _return_ = env.sell(portfolio[share_name], share_name)
                portfolio[share_name] = 0
                budget = budget + _return_
        len_buy = action.count(0)
        buy_budget = budget
        if len_buy != 0:
            buy_budget = budget/len_buy #if we want to buy several shares
        for share_name in self.shares_dict:
            a = action[self.shares_dict[share_name]]
            a_string = self.actions_dict[a]
            if a_string=='buy':
                amount, change = env.buy(buy_budget, share_name)
                portfolio[share_name] += amount
                budget = budget - buy_budget + change

        return [budget, portfolio], self._get_reward(budget, portfolio)

    def action_value_estimation(self, state):
        x = self.features_generator(self.current_state())
        estimated_action_values = self.model.predict([x[0], x[1]])
        return estimated_action_values[0]

    @staticmethod
    def encode(action_index):
        action = []
        action.append(int(action_index/3**2))
        action_index = action_index%3**2
        action.append(int(action_index/3**1))
        action_index = action_index%3**1
        action.append(action_index)
        return action

    @staticmethod
    def decode(action):
        action_index = action[0]*3**2 + action[1]*3**1 + action[2]*3**0
        return action_index

    def evaluate(self):
        self.funds_reset()
        #baseline
        #sell all and buy evenly all
        action = [0, 0, 0]
        baseline_state = self.get_new_state(self.current_state(), action)[0]
        #raise Exception
        # evaluation
        while self.env.testlimit - self.env.timestamp > 0:
            self.wait_till_prices_update()
            action_values = self.action_value_estimation(self.current_state())
            choosen_action = np.argmax(action_values)
            action = self.encode(choosen_action)
            new_state = self.get_new_state(self.current_state(), action)[0]
            self.set_state(new_state)
        return self.get_new_state(self.current_state(), [2,2,2]), self.get_new_state(baseline_state,[2,2,2])

    def train(self, epsilon=0.7):
        terminal_flag = False
        while not terminal_flag:
            states_returns_actions = []
            self.wait_till_prices_update()
            action_values = self.action_value_estimation(self.current_state())
            # if self.env.timestamp%100==0:
            #     print(self.env.timestamp)

            if self.env.trainlimit - self.env.timestamp > 0:
                if np.random.rand() > p:
                    choosen_action = np.argmax(action_values)
                else:
                    choosen_action = np.random.randint(0, len(action_values))
                for a_0 in range(3):
                    for a_1 in range(3):
                        for a_2 in range(3):
                            action = [a_0, a_1, a_2]
                            action_index = self.decode(action)
                            if action_index == choosen_action:
                                new_state, reward = self.get_new_state(self.current_state(), action)
                                max_estimated_return = max(self.action_value_estimation(new_state))
                                return_ = action_values[action_index] + ALPHA*(reward + GAMMA*(max_estimated_return - action_values[action_index]))
                            else:
                                return_ = action_values[action_index]
                            states_returns_actions.append((self.current_state(), self.return_scaler.partial_fit([[return_]]).transform([[return_]])[0][0] , action_index))
            else: #if we in terminal state sell all
                terminal_flag = True
                for a_0 in range(3):
                    for a_1 in range(3):
                        for a_2 in range(3):
                            action = [2, 2, 2]
                            action_index = self.decode(action)
                            new_state, reward = self.get_new_state(self.current_state(), action)
                            max_estimated_return = max(self.action_value_estimation(new_state))
                            return_ = action_values[action_index] + ALPHA*(reward + GAMMA*(max_estimated_return - action_values[action_index]))
                            states_returns_actions.append((self.current_state(), self.return_scaler.partial_fit([[return_]]).transform([[return_]])[0][0] , action_index))

            x_batch = [self.features_generator(state[0]) for state in states_returns_actions]
            x_batch = [[x[0][0] for x in x_batch], [x[1][0] for x in x_batch]]
            y_batch = [action[2] for action in states_returns_actions]
            y_batch = keras.utils.to_categorical(y_batch)
            y_batch = np.multiply(y_batch, [return_[1] for return_ in states_returns_actions])
            self.model.train_on_batch(x_batch, y_batch)
            self.set_state(new_state)

        print(self.return_scaler.mean_)

env = Environment(prices)
agent = Agent(env)
for i in range(300):
    print(i)
    agent.train()
    print(agent.evaluate())
    env.reset()
    agent.reset()

# print(prices.head())
# print(env.current_prices())
# print(env.sell(2, 'AAPL'))
