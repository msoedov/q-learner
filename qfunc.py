import argparse
import logging
import pickle
import random
import sys
from collections import defaultdict

import numpy as np

import gym
from gym import wrappers


class QFunc:
    """
    Q-Learner
    gamma_factor -  discount coef
    (1 - epsilon) - exploration probability
    exploration_decay - exploration decay on each action
    q_table - history of observed states  nested hashtable state => action => reward
    """

    learning_rate = 0.1
    gamma_factor = 0.9
    epsilon = 0.1
    exploration_decay = 1.00005

    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = defaultdict(dict)

    def _hash_word_state(self, state: np.array) -> int:
        """
        Hashes word state of multy dim np.array into an int. Will result a low memory footprint for
        hash table
        """

        return hash(tuple(state.flatten()))

    def size(self) -> int:
        return len(self.q_table)

    def learn(self,
              old_state: np.array,
              action: int,
              reward: int,
              new_state: np.array) -> None:
        """
        Definition of Q-learning taken from

        https://en.wikipedia.org/wiki/Q-learning
        """
        q_old_state = self.q_table[self._hash_word_state(old_state)].get(
            action, random.randint(1, 10))  # use random to init table if value does not exists
        q_new_state_max = max(self.q_table[self._hash_word_state(new_state)] or
                              [random.randint(1, 10)])
        self.q_table[self._hash_word_state(old_state)][action] = q_old_state + self.learning_rate * \
            (reward + self.gamma_factor * q_new_state_max - q_old_state)

    def make_decision(self, state: np.array) -> None:
        """
        Decides which action to take.

        Args:
         	state: current board state
        """

        self.epsilon = self.exploration_decay * self.epsilon
        if random.random() > self.epsilon:
            return self.action_space.sample()
        else:
            action = max(self.q_table[self._hash_word_state(state)] or
                         [self.action_space.sample()])
            return action


def train():
    env = gym.make('SpaceInvaders-v0')
    outdir = '/tmp/q-space-func'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    q_learner = QFunc(env.action_space)
    episode_count = 1000
    reward = 0
    done = False
    max_score = 0
    all_time_max = 0
    for i in range(episode_count):
        state = env.reset()
        print('#' * 50)
        print("Current score", max_score)
        print("Max score", all_time_max)
        print("Game number #", i)
        print("Observed states", q_learner.size())
        print("Exploration factor {:.1f}%".format((1 - q_learner.epsilon) *
                                                  100))
        print("Q function decisions factor {:.1f}%".format((q_learner.epsilon)
                                                           * 100))
        all_time_max = max(all_time_max, max_score)
        max_score = 0
        while True:
            action = q_learner.make_decision(state)
            state_ = state
            state, reward, done, _ = env.step(action)
            q_learner.learn(
                old_state=state_,
                action=action,
                reward=reward,
                new_state=state)
            max_score += reward
            if done:
                break
            # env.render()
    with open('qfunc.pickle', 'wb') as handle:
        pickle.dump(q_learner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    env.close()


if __name__ == '__main__':
    train()
