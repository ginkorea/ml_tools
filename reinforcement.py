import numpy as np
import pandas as pd
import copy


class Action:

    """defines possible actions"""

    def __init__(self):
        self.actions = ['-1,-1', '-1,0', '-1,1', '0,-1', '0,0', '0,1', '1,-1', '1,0', '1,1']

    def select_action_column(self, x_acc, y_acc):
        """get column index based on x,y acc"""
        index = -np.inf
        for i, acts in enumerate(self.actions):
            pair = acts.split(",")
            if (int(pair[0]) == x_acc) and (int(pair[1]) == y_acc):
                index = i
                break
        return index


class Table:

    """the Q table"""

    def __init__(self, track):
        possible_actions = Action()  # actions = columns
        self.actions = possible_actions.actions
        self.speeds = self.determine_possible_speeds()  # calculates all possible speeds (-5,-5) -> (5,5)
        self.states = self.define_states(self.speeds, track)  # calculates all the states based on loc / speed combos
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.zeros = self.create_zeros(self.n_states, self.n_actions)  # init chart to zeros
        self.title = track.name
        self.track = track
        self.df = pd.DataFrame(data=self.zeros, columns=self.actions, index=self.states)  # create table

    def load(self, filename):

        """load previously learned q-table"""

        self.df = pd.read_csv(filename, index_col=0)

    def determine_possible_speeds(self, low=-5, high=5):

        """returns possible speeds to help build states"""

        pairs = []
        for x in range(low, high + 1):
            for y in range(low, high + 1):
                speed = str(x) + "," + str(y)
                pairs.append(speed)
        self.speeds = pairs
        return pairs

    def define_states(self, pos_speeds, track):

        """determines all possible states for the track"""

        course_tiles = []
        possible_states = []
        for line in track.matrix:
            for tile in line:
                if tile.course:
                    course_tiles.append(tile)
        for tile in course_tiles:
            for speed in pos_speeds:
                state = tile.id + "|" + speed
                possible_states.append(state)
        self.states = possible_states
        return possible_states

    def create_zeros(self, rows, columns):
        zeros = np.zeros([rows, columns])
        self.zeros = zeros
        return zeros

    def select_state(self, x_pos, y_pos, x_speed, y_speed):

        """selects state based on pos and speed"""

        tile = self.track.get_tile(x_pos, y_pos)
        state = tile.id + "|" + str(x_speed) + "," + str(y_speed)
        return state

    def select_action(self, x_acc, y_acc):

        """selects action based on x,y acc"""

        action = str(x_acc) + "," + str(y_acc)
        return action

    def fetch(self, state, action):
        return self.df.at[state, action]


class ReinforcementLearner:

    """base class for a reinforcement learner, inherited by q and sarsa"""

    def __init__(self, car, mode=None, discount_rate=.9, learning_rate=1, epsilon=.5):
        self.discount = discount_rate
        self.learning = learning_rate
        self.epsilon = epsilon
        self.track = car.track
        self.car = car
        self.table = self.track.table
        self.actions = self.table.actions
        self.mode = mode.lower()  # q or sarsa
        self.q1 = None
        self.q2 = None
        self.selected = None
        self.follow_on = None
        self.reward = None
        self.predetermined = False

    def __str__(self):
        string = "%s learner on %s" % (self.mode, self.track.name)
        return string

    def determine_action(self, car):
        """determines the action based on policy"""
        car.set_state_action_pair()
        random = np.random.rand(1)
        if random > self.epsilon:
            q_value, index = self.get_best_action()
        else:
            index = np.random.randint(0, len(self.actions))
        action = self.actions[index]
        return action

    def get_best_action(self, trial=False, evaluate=None):
        """returns the state/action pair with highest q-value"""
        best_q = -np.infty
        possible_q = []
        q_index = None
        if not trial:
            evaluate = self.track.table.df.loc[self.car.state]
        for i, q_value in enumerate(evaluate):
            if q_value > best_q:
                best_q = q_value
                q_index = i
                possible_q = [i]
            elif q_value == best_q:
                possible_q.append(i)
        if len(possible_q) > 1:
            random = np.random.randint(0, len(possible_q))
            q_index = possible_q[random]
        return best_q, q_index

    def determine_follow_on_action(self):
        pass

    def adjust_policy(self, percent=10):
        """not used in program"""
        adjusted_epsilon = self.epsilon + self.epsilon * (percent / 100)
        if 1 > adjusted_epsilon > 0:
            self.epsilon = adjusted_epsilon
        elif adjusted_epsilon > 1:
            self.epsilon = 1
        elif adjusted_epsilon < 0:
            self.epsilon = 0
        return self.epsilon

    def get_reward(self, car):
        """gets the reward from its current position, sub 10 for crashing"""
        x = car.position.x
        y = car.position.y
        tile = self.track.get_tile(x, y)
        reward = tile.reward
        if car.crashed:
            reward = reward - 10  # because we are not actually on the out tile
        elif car.finished:
            reward = reward + 100  # extra bonus points makes finishing actually worth 200
        return reward

    def store_state_action(self, car, initial=True):
        """stores the state / action pair from the car into the learner"""
        if initial:
            self.selected = [car.state, car.action]
        else:
            self.follow_on = [car.state, car.action]

    def update_q(self):
        """q-update policy, based on q-learning. overwritten in sarsa"""
        # current_q = self.table.df.loc[self.selected[0], self.selected[1]]
        # current_q += self.learning * (self.reward + (self.discount * self.q2) - self.q1)
        current_q = self.learning * (self.reward + (self.discount * self.q2))
        self.table.df.loc[self.selected[0], self.selected[1]] = current_q


class Q(ReinforcementLearner):

    """class for q-learner"""

    def __init__(self, car, mode="q", discount_rate=.9, learning_rate=1, epsilon=.5):
        super().__init__(car, mode="q", discount_rate=.9, learning_rate=1, epsilon=.5)

    def determine_follow_on_action(self):
        """determine follow-on action based on q method"""
        ghost_car = copy.copy(self.car)  # so we don't move there create a copy
        ghost_car.crashed = False
        ghost_car.update_state()
        action = ghost_car.learner.determine_action(ghost_car, ghost=True)  # gets best action
        ghost_car.update_action(action)
        state = ghost_car.state
        # ghost_car.accelerate(action)
        q2 = ghost_car.get_current_state_action_pair_q()  # highest q
        return q2, state, action

    def determine_action(self, car, ghost=False, course=None):
        """determine action based on q method"""
        # car.set_state_action_pair()
        random = np.random.rand(1)
        if random > self.epsilon or ghost:  # ghost gets the best just like exploiting
            q_value, index = self.get_best_action()
        else:
            index = np.random.randint(0, len(self.actions))  # explore
        action = self.actions[index]
        return action


class SARSA(ReinforcementLearner):

    """the sarsa-learner"""

    def __init__(self, car, mode="sarsa", discount_rate=.9, learning_rate=.33, epsilon=.5):
        super().__init__(car, mode="sarsa", discount_rate=.9, learning_rate=.33, epsilon=.5)

    def determine_follow_on_action(self):
        """determine follow-on action and q-value using sarsa method"""
        ghost_car = copy.copy(self.car)
        ghost_car.crashed = False
        ghost_car.update_state()
        action = ghost_car.learner.determine_action(ghost_car, ghost=True)
        ghost_car.update_action(action)
        state = ghost_car.state
        # ghost_car.accelerate(action)
        q2 = ghost_car.get_current_state_action_pair_q()
        self.predetermined = True  # the learner will follow the predetermined action in next turn
        return q2, state, action

    def determine_action(self, car, ghost=False):
        """determine action based on sarsa method"""
        # car.set_state_action_pair()
        random = np.random.rand(1)
        if random > self.epsilon:  # follows normal epsilon greedy policy
            q_value, index = self.get_best_action()
        else:
            index = np.random.randint(0, len(self.actions))  # explore
        action = self.actions[index]
        return action

    def update_q(self):
        """update q-table based on sarsa update method"""
        current_q = self.table.df.loc[self.selected[0], self.selected[1]]
        current_q += self.learning * (self.reward + (self.discount * self.q2) - self.q1)
        # current_q = self.learning * (self.reward + (self.discount * self.q2))
        self.table.df.loc[self.selected[0], self.selected[1]] = current_q
