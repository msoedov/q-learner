import numpy
import random
import tflearn
import argparse
import logging
import pickle
import random
import sys
import cv2
from collections import defaultdict
from sklearn.preprocessing import normalize
import numpy as np
import better_exceptions
import gym
from gym import wrappers
import tensorflow as tf


class Model:
    def __init__(self, net, layer):
        self.net = net
        self.layer = layer

    @classmethod
    def new(cls) -> tflearn.DNN:
        net = tflearn.input_data(shape=[None, 84, 84, 1], name='space')
        hidden = net = tflearn.fully_connected(net, 6, activation='relu')
        net = tflearn.regression(
            net, optimizer='adam', learning_rate=0.01, name='target')
        m = tflearn.DNN(net)
        print(m.get_weights(hidden.W).shape)
        print(m.get_weights(hidden.b).shape)
        return cls(m, hidden)


class Genome:
    def __init__(self, model, weights, bias):
        self.weights = weights
        self.bias = bias
        self.net = self.model = model

    @classmethod
    def new(cls, model) -> tflearn.DNN:
        w = numpy.random.rand(7056, 6)
        b = numpy.random.rand(6)
        return cls(model, w, b)

    def start(self):
        self.model.net.set_weights(self.model.layer.W, self.weights)
        self.model.net.set_weights(self.model.layer.b, self.bias)

    def end(self):
        self.weights = self.model.net.get_weights(self.model.layer.W)
        self.bias = self.model.net.get_weights(self.model.layer.b)

    def clone(self):
        m = self.new(self.model)
        m.weights = self.weights
        m.bias = self.bias
        m.mutate()
        return m

    def mutate(self) -> tflearn.DNN:
        weights = self.weights
        dim = int(weights.shape[0] * weights.shape[1] * 0.001)
        for _ in range(dim):
            col, row = random.randint(0, weights.shape[0] - 1), random.randint(
                0, weights.shape[1] - 1)
            new_val = weights[col][row] + random.uniform(-1, 1)
            weights[col][row] = min(max(new_val, -1), 1)
            self.bias[row] += random.uniform(-1, 1)
        return weights

    def cross_over(self, other):
        m = self.new(self.model)
        weights = self.weights
        other_weights = other.weights
        for i, r in enumerate(weights):
            if i % 2 == 0:
                continue
            weights[i] = other_weights[i]
        m.weights = weights

        bias = self.bias
        other_bias = other.bias
        for i, r in enumerate(bias):
            if i % 2 == 0:
                continue
            bias[i] = other_bias[i]
        m.bias = bias
        return m

    def make_decision(self, state: np.array) -> int:
        state = preprocess(state)
        feed = numpy.array(state).reshape(1, 84, 84, 1)
        actions = self.model.net.predict(feed)
        choosen = numpy.argmax(actions[0])
        return choosen


class Pool:
    def __init__(self, size=24):
        self.net = Model.new()
        self.size = size
        self.pool = [Genome.new(self.net) for _ in range(self.size)]
        self.scores = []
        self.gen = 0

    def score(self, gen, score):
        self.scores.append((score, gen))

    def next_gen(self):
        print('=' * 50)
        self.gen += 1
        print('Generation', self.gen)
        fit_num = int(self.size * 0.2 + 0.5)
        top3 = sorted(self.scores, key=lambda x: x[0])[-fit_num:]
        print(top3)
        new_pool = []
        for j in range(int(self.size / fit_num)):
            child = [
                top3[-1 - j][1].cross_over(top3[-i][1]).clone()
                for i in range(fit_num)
            ]
            new_pool.extend(child)
        self.pool = new_pool
        self.scores = []


def preprocess(observation: np.array) -> np.array:
    """
    Transform 265 x 160 x 3 -> 84 x 84 x 1 world
    """

    observation = cv2.cvtColor(
        cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))


def train():
    env = gym.make('SpaceInvaders-v0')
    outdir = '/tmp/q-space-func'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    episode_count = 100
    reward = 0
    done = False
    max_score = 0
    pool = Pool()
    all_time_max = 0
    for i in range(episode_count):
        for m in pool.pool:
            m.start()
            state = env.reset()
            if max_score > all_time_max:  # Enable recording
                pool.net.net.save("savepoint/model.tfl".format(i, max_score))
            all_time_max = max(all_time_max, max_score)
            print('Gen', i)
            print('#' * 50)
            print("Current score", max_score)
            print("Max score", all_time_max)
            max_score = 0
            while True:
                action = m.make_decision(state)
                state_ = state
                state, reward, done, info = env.step(action)
                max_score += reward
                if done:
                    break
                env.render()
            m.end()
            pool.score(m, max_score)
        pool.next_gen()

    env.close()

    env = wrappers.Monitor(env, directory=outdir, force=True)
    m = pool.pool[0]
    m.start()
    state = env.reset()
    while True:
        action = m.make_decision(state)
        state_ = state
        state, reward, done, info = env.step(action)
        max_score += reward
        if done:
            break
        env.render()
    m.end()
    env.close()


if __name__ == '__main__':
    train()
