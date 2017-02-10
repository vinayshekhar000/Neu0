import numpy as np

import math


class RegisterBank:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.memory = [0] * size

    def read_fuzzy(self, weights):
        '''
        :param weights: numpy array with weights for each cell
        :return: value read fuzzily
        '''
        weights = weights.tolist()
        value = 0
        for weight, cell_value in zip(weights, self.memory):
            value += weight * cell_value
        return value

    def read(self, index):
        return self.memory[index]

    def write(self, index, value):
        self.memory[index] = value

    def get_memory(self):
        return self.memory

    def write_fuzzy(self, weights, value):
        """
        :param weights:  numpy array with weights for each cell
        :param value: value to write to each cell according to weights
        :return: None
        """
        weights = weights.tolist()
        # retain
        retain_weights = [1 - x for x in weights]
        # We do not need erase vector
        for index in range(len(self.memory)):
            self.memory[index] = retain_weights[index] * self.memory[index]
        # Add
        for index in range(len(self.memory)):
            self.memory[index] += weights[index] * value


class MemoryBank:
    def __init__(self, name, size, variance=0.05):
        self.size = size
        self.name = name
        self.memory = [0] * size
        self.variance = variance

    def make_gaussian(self, mean):
        scale = 1 / math.sqrt(2 * math.pi * self.variance)
        return lambda x: scale * math.e ** -((x - mean) ** 2 / (2 * self.variance))

    def get_threshold(self, gauss, value):
        lower = int(value)
        while (gauss(lower) > 0):
            lower -= 1
        return int(value) - lower

    def read(self,index):
        return self.memory[index]

    def write(self,index,value):
        self.memory[index] = value

    def read_gauss(self, value):
        gauss = self.make_gaussian(value)
        thresh = self.get_threshold(gauss, value)
        range_ = (int(value) - thresh, int(value) + thresh)
        range_ = (max(range_[0], 0), min(range_[1], self.size))
        probs = {}  # * self.size
        sum_ = 0
        for mem in range(range_[0], range_[1]):
            probs[mem] = math.e ** gauss(mem)
            sum_ += probs[mem]
        for mem in range(range_[0], range_[1]):
            probs[mem] = probs[mem] / sum_
        sum_ = 0
        for x in range(range_[0], range_[1]):
            probs[x] = math.e ** (math.log(probs[x]) / 0.05)
            sum_ += probs[x]
        for mem in range(range_[0], range_[1]):
            probs[mem] = probs[mem] / sum_

        value = 0
        for mem in range(range_[0], range_[1]):
            value += self.memory[mem] * probs[mem]
        return value

    def write_gauss(self, value, content):
        gauss = self.make_gaussian(value)
        thresh = self.get_threshold(gauss, value)
        range_ = (int(value) - thresh, int(value) + thresh)
        range_ = (max(range_[0], 0), min(range_[1], self.size))
        probs = {}
        sum_ = 0
        for mem in range(range_[0], range_[1]):
            probs[mem] = math.e ** gauss(mem)
            sum_ += probs[mem]
        for mem in range(range_[0], range_[1]):
            probs[mem] = probs[mem] / sum_
        sum_ = 0
        for x in range(range_[0], range_[1]):
            probs[x] = math.e ** (math.log(probs[x]) / 0.05)
            sum_ += probs[x]
        for mem in range(range_[0], range_[1]):
            probs[mem] = probs[mem] / sum_
        for mem in range(range_[0], range_[1]):
            self.memory[mem] = (1 - probs[mem]) * self.memory[mem]
        for mem in range(range_[0], range_[1]):
            self.memory[mem] += probs[mem] * content


    def get_memory(self):
        return self.memory
