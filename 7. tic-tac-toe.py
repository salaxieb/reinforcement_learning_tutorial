import numpy as np

EPSILON_GREEDY = 0.9
ALPHA_DISCOUNT = 0.9

class Environment():
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height
        self.reset_state()

    def draw(self):
        print('-'*4)
        for i in range(self.height):
            print(self.current_state[self.width*i : self.width*i+4])
        print('-'*4)

    def reset_state(self):
        self.current_state = '_'*self.width*self.height

    def is_draw(self, state):
        return np.all([c != '_' for c in state])

    def is_win(self, state):
        for i in range(self.width):
            if (state[self.width*i] == state[self.width*i + 1]
             and state[self.width*i + 1] == state[self.width*i + 2]
              and state[self.width*i + 2] == state[self.width*i + 3]
               and state[self.width*i]!='_'):
                # print('full row')
                return state[self.width*i]

        for i in range(self.width):
            if (state[i + self.width*0] == state[i + self.width*(1)]
             and state[i + self.width*(1)] == state[i + self.width*(2)]
              and state[i + self.width*(2)] == state[i + self.width*(3)]
               and state[i+self.width*0]!='_'):
                # print('full column')
                return state[i + self.width*0]

        if (state[0 + self.width*0] == state[1 + self.width*1]
         and state[1 + self.width*1] == state[2 + self.width*2]
          and state[2 + self.width*2] == state[3 + self.width*3]
           and state[0 + self.width*0]!='_'):
            # print('diag')
            return state[0 + self.width*0]

        if (state[3 + self.width*0] == state[2 + self.width*1]
         and state[2 + self.width*1] == state[1 + self.width*2]
          and state[1 + self.width*2] == state[0 + self.width*3]
           and state[3 + self.width*0]!='_'):
            # print('rev diag')
            return state[3 + self.width*0]
        return False

    def values_initializer(self, state, player_figure):
        if self.is_draw(state):
            return 0.1
        winner_figure = self.is_win(state)
        if winner_figure:
            if winner_figure == player_figure:
                return 1
            else:
                return 0
        # if not won not not lost
        return 0.5

    def game_over(self):
        # print('is win', self.is_win(self.current_state))
        # print('is draw', self.is_draw(self.current_state))
        if self.is_win(self.current_state) or self.is_draw(self.current_state):
            return True
        else:
            return False

    def action_made(self, action, player):
        assert self.current_state[action] == '_'
        self.current_state = ''.join([self.current_state[i] if i!=action else player.played_figure for i in range(len(self.current_state))])

class Agent():
    def __init__(self, played_figure):
        self.played_figure = played_figure
        self.estimated_values = {}
        self.episode_states = []
        self.wins = 0
        self.verbose = False

    def give_value_for_next_state(self, env, state, action):
        state = self.new_state_after_action(state, action)
        if state in self.estimated_values:
            return self.estimated_values[state]
        else:
            self.estimated_values[state] = env.values_initializer(state, self.played_figure)
        return self.estimated_values[state]

    def new_state_after_action(self, state, action):
        state = ''.join([state[i] if i!=action else self.played_figure for i in range(len(state))])
        return state

    def choose_action(self, env):
        possible_actions = self.give_possible_actions(env)
        if self.verbose:
            print('possible_actions', possible_actions)
        if np.random.rand() < EPSILON_GREEDY:
            action = np.random.choice(possible_actions)
        else:
            maxV_action = possible_actions[0]
            maxV = self.give_value_for_next_state(env, env.current_state, maxV_action)
            for action in possible_actions:
                if self.verbose:
                    print(action, self.give_value_for_next_state(env, env.current_state, action))
                if self.give_value_for_next_state(env, env.current_state, action) > maxV:
                    maxV_action = action
                    maxV = self.give_value_for_next_state(env, env.current_state, maxV_action)
            action = maxV_action
        self.give_value_for_next_state(env, env.current_state, action)
        self.episode_states.append(self.new_state_after_action(env.current_state, action))
        return action

    def give_possible_actions(self, env):
        possible_actions = []
        for i in range(len(env.current_state)):
            if env.current_state[i] == '_':
                possible_actions.append(i)
        return possible_actions

    def update_estimated_values(self, reward):
        self.wins += reward
        target = reward
        if self.verbose:
            print('episode states', self.episode_states)
        for state in reversed(self.episode_states):
            if self.verbose:
                print('before', state, self.estimated_values[state])
            self.estimated_values[state] += ALPHA_DISCOUNT * (target - self.estimated_values[state])
            target = self.estimated_values[state]
            if self.verbose:
                print('after', state, self.estimated_values[state])
        self.episode_states = []

class Human():
    def __init__(self, played_figure):
        self.played_figure = played_figure
        self.verbose = True

    def choose_action(self, env):
        if self.verbose:
            env.draw()
        action = input("make your smart move, bag with bones ")
        return int(action) - 1

    def update_estimated_values(*args):
        pass


def play_game(agentX, agentO, env):
    env.reset_state()
    queue = {agentX: agentO,
             agentO: agentX}
    player = agentX
    while not env.game_over():
        action = player.choose_action(env)
        if player.verbose:
            print(player.played_figure, action)
        env.action_made(action, player)
        player = queue[player]

    winner_figure = env.is_win(env.current_state)
    if winner_figure and winner_figure == agentX.played_figure:
        agentX.update_estimated_values(1)
        agentO.update_estimated_values(0)
    elif winner_figure and winner_figure == agentO.played_figure:
        agentX.update_estimated_values(0)
        agentO.update_estimated_values(1)
    elif env.is_draw(env.current_state):
        agentX.update_estimated_values(0.3)
        agentO.update_estimated_values(0.3)
    else:
        print("why stopped!?")
    if agentX.verbose:
        env.draw()
    return winner_figure


agentX = Agent(played_figure='X')
agentO = Agent(played_figure='O')
env = Environment()

for i in range(1, 3000000):
    if i%1000==0:
        print(i)
    EPSILON_GREEDY = 0.1#1/(1.01**i)
    play_game(agentX, agentO, env)
print(len(agentX.estimated_values))
print('X', agentX.wins, 'O', agentO.wins)
agentX.verbose = True
while True:
    EPSILON_GREEDY = 0
    play_game(agentX, agentO, env)
    play_game(agentX, agentO, env)
    play_game(agentX, agentO, env)
    print('last_winner', play_game(agentX, Human('O'), env))
