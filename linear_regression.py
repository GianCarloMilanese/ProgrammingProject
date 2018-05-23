import numpy as np
from sklearn import datasets

boston = datasets.load_boston()
features = boston.data
target = boston.target


class BostonLinearRegression(object):

    def __init__(self, initial_parameters=None, learning_rate=0.001):
        self.num_features = len(features[0])
        self.num_examples = len(features)
        self.initial_parameters = np.array(initial_parameters)
        m = np.zeros((self.num_examples, self.num_features+1))+1
        m[:, 1:] = features
        self.features = m
        self.target = target
        self.learning_rate = learning_rate

    def h_function(self, example):
        return np.dot(self.initial_parameters, self.features[example])

    def cost(self):
        my_list = []
        for i in range(self.num_examples):
            my_list.append((self.h_function(i)-self.target[i])**2)
        s = sum(my_list)
        return (1/(2*self.num_examples))*s

    def cost_der(self, parameter_index):
        my_list = []
        for i in range(self.num_examples):
            my_list.append((self.h_function(i)-self.target[i])*self.features[i][parameter_index])
        return sum(my_list)

    def update_parameters(self):
        new_parameter_list = []
        for i in range(len(self.initial_parameters)):
            new_parameter_list.append(self.initial_parameters[i] -
                                      self.learning_rate*(1/self.num_examples)*self.cost_der(i))
        self.initial_parameters = np.array(new_parameter_list)

    def gradient_descent(self, repetitions):
        for i in range(repetitions):
            print('before:', self.initial_parameters)
            self.update_parameters()
            print('after:', self.initial_parameters)


a = BostonLinearRegression([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
print(a.features[0])
print(features[0])
print('h:', a.h_function(0))
print(a.initial_parameters*a.features[0])
print(sum(a.initial_parameters*a.features[0]))
print(a.cost())
print(a.cost_der(1))
print(a.features[500][0])
print(a.initial_parameters)
a.update_parameters()
print(a.initial_parameters)

#a.gradient_descent(1000)
