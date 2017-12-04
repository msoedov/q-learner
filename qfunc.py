import argparse
import logging
import pickle
import random
import sys
import cv2
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

    q_hits = 0
    all_hits = 1
    action_variety = defaultdict(int)

    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = defaultdict(dict)

    def _hash_word_state(self, state: np.array) -> int:
        """
        Hashes word state of multy dim np.array into an int. Will result a low memory footprint for
        hash table
        """
        # return hash(tuple(reduce_world(state).flatten()))
        return reduce_state_2d(state=state)

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
            action, random.randint(0, 2))
        q_new_state_max = max(self.q_table[self._hash_word_state(new_state)] or
                              [random.randint(0, 2)])
        val = (1 - self.learning_rate) * q_old_state + self.learning_rate * \
            (reward + self.gamma_factor * q_new_state_max)
        self.q_table[self._hash_word_state(old_state)][action] = val

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
            if self.q_table[self._hash_word_state(state)]:
                self.q_hits += 1
            self.all_hits += 1
            key = self._hash_word_state(state)
            if self.q_table[key]:
                action = max(self.q_table[key],
                             key=lambda key: self.q_table.get(key, 0))
            else:
                action = self.action_space.sample()
            self.action_variety[action] += 1
            return action

    def hit_ratio(self) -> float:
        return self.q_hits / self.all_hits

    def exploration_factor(self) -> float:
        return min((1 - self.epsilon) * 100, 100)


def reduce_world(data, rows=60, cols=120, reshape=True):
    if reshape:
        data = data.reshape([210, 160 * 3])
    row_sp = data.shape[0] // rows
    col_sp = data.shape[1] // cols
    tmp = np.sum(data[i::row_sp] for i in range(row_sp))
    return np.sum(tmp[:, i::col_sp] for i in range(col_sp))


def preprocess(observation: np.array) -> np.array:
    """
    Transform 265 x 160 x 3 -> 84 x 84 x 1 world
    """

    observation = cv2.cvtColor(
        cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))


def reduce_state_2d(state):
    """
    Dummy method to detect positions of ship, bullet and enemy ships
    """
    ship = preprocess(state)[68:75]
    obstacles = preprocess(state)[50:55]
    enemy = preprocess(state)[40:50]
    obstacles_arg = np.argmax(reduce_world(obstacles, 1, 84, reshape=False).T)
    ship_arg = np.argmax(reduce_world(ship, 1, 84, reshape=False).T)
    enemy_arg = np.argmax(reduce_world(enemy, 1, 84, reshape=False).T)
    return ship_arg, enemy_arg, obstacles_arg


def snaphot(world):
    import matplotlib.pylab as plt
    plt.imshow(np.array(np.squeeze(world)))
    plt.show()


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
        print("Exploration probability {:.1f}%".format(
            q_learner.exploration_factor()))
        print("Qs memory hit", q_learner.hit_ratio())
        print("Action Variety", q_learner.action_variety)
        q_learner.action_variety = defaultdict(int)
        all_time_max = max(all_time_max, max_score)
        max_score = 0
        lives = 3
        while True:
            action = q_learner.make_decision(state)
            state_ = state
            state, reward, done, info = env.step(action)
            penalty = 0
            new_lives = info['ale.lives']
            if new_lives < lives:
                lives = new_lives
                penalty = 20
            q_reward = reward - penalty
            q_learner.learn(
                old_state=state_,
                action=action,
                reward=q_reward,
                new_state=state)
            max_score += reward
            if done:
                break
            if q_learner.epsilon > 0.5:
                env.render()
    with open('qfunc.pickle', 'wb') as handle:
        pickle.dump(q_learner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    env.close()


if __name__ == '__main__':
    train()
