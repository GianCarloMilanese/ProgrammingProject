import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import math


class RegularizedLinearRegression(object):

    def __init__(self, features=np.array([]), target=np.array([])): #maybe no need for empty array
        self.num_features = len(features[0]) #len of first row
        self.num_examples = len(features) # = 506
        self.initial_parameters = np.array([1 for i in range(self.num_features + 1)]) # array of 1's
        m = np.zeros((self.num_examples, self.num_features+1))+1 #matrix of all 1s: 506*14
        m[:, 1:] = features
        self.features = m #not number of features, just a placeholder
        self.target = target

    "The hypothesis function = theta^t * x"
    def h_function(self, example):
        return np.dot(self.initial_parameters, self.features[example])

    "The Cost function J(theta)=1/2m*sum_{i=1^m}(h_theta(x^i)-y^i)^2"
    def cost(self, lamb):
        my_list = []
        for i in range(self.num_examples):
            my_list.append((self.h_function(i)-self.target[i])**2)
        s = sum(my_list)
        parameter_list = []
        for i in range(1, self.num_features):
            parameter_list.append(self.initial_parameters[i]**2)
        ss = sum(parameter_list)
        return (1/(2*self.num_examples))*(s + lamb*ss)

    def cost_der(self, parameter_index, lamb):
        #not sure if parameter_index needs to be in there?
        """
        returns the derivative of the cost function wrt the parameter with index parameter_index
        :param parameter_index: the index of the parameter
        :return: derivative with respect to the specified parameter of the cost function
        """
        my_list = []
        for i in range(self.num_examples):
            my_list.append((self.h_function(i)-self.target[i])*self.features[i][parameter_index])
        return (sum(my_list))/self.num_examples + (lamb*self.initial_parameters[parameter_index])/self.num_examples

    def update_parameters(self, learning_rate=0.01, lamb=1000):
        new_parameter_list = [] #simultaneous update of parameters
        for i in range(len(self.initial_parameters)):
            new_parameter_list.append(self.initial_parameters[i] -
                                      learning_rate*self.cost_der(i, lamb))
        self.initial_parameters = np.array(new_parameter_list)

    def gradient_descent(self, repetitions, learning_rate, lamb):
        cost_list = []
        for i in range(repetitions):
            c = self.cost(lamb)
            cost_list.append(self.cost(lamb))
            self.update_parameters(learning_rate)
            if i%(repetitions/5) == 0:
                plt.plot([feature[1] for feature in self.features], self.target, color='g', linewidth=0, marker='o')
                xs = np.linspace(0, max(np.hstack(self.features)), 100)
                theta0 = self.initial_parameters[0]
                theta1 = self.initial_parameters[1]
                ys = [theta0 + theta1*x for x in xs]  # this is the h function
                plt.plot(xs, ys, color='r')
                plt.show()
            if c - self.cost(lamb) < 0.0001:
                print('convergence at iteration', i)
        cost_array = np.array(cost_list)
        plt.plot(range(repetitions), cost_array)
        plt.show()

    def gradient_descent2(self, learning_rate, lamb):
        repetitions = 0
        cost_list = []
        while True:
            cost_before = self.cost(lamb)
            cost_list.append(self.cost(lamb))
            self.update_parameters(learning_rate, lamb)

            if repetitions%500 == 0:   # this plots the hypothesis function; works for the basic case with one feature
                for i in range(1, len(self.features[0])):
                    plt.plot([feature[i] for feature in self.features], self.target, color='g', linewidth=0, marker='o')
                    theta0 = self.initial_parameters[0]
                    theta1 = self.initial_parameters[i]
                    xs = np.linspace(min(self.features[:, i]), max(self.features[:, i]), 100)
                    ys = [theta0 + theta1 * x for x in xs]  # this is the h function
                    plt.plot(xs, ys, color='r')
                    plt.title('Iteration {}, feature number {}'.format(repetitions, i))
                    plt.show()
            repetitions += 1
            #print('cost:', self.cost(lamb))
            if np.abs(cost_before - self.cost(lamb)) < 0.0001:  # tests convergence
                for i in range(1, len(self.features[0])):
                    plt.plot([feature[i] for feature in self.features], self.target, color='g', linewidth=0, marker='o')
                    theta0 = self.initial_parameters[0]
                    theta1 = self.initial_parameters[i]
                    xs = np.linspace(min(self.features[:, i]), max(self.features[:, i]), 100)
                    ys = [theta0 + theta1 * x for x in xs]  # this is the h function
                    plt.plot(xs, ys, color='r')
                    plt.title('Convergence at iteration {}, feature number {}'.format(repetitions, i))
                    plt.show()
                break
        cost_array = np.array(cost_list)
        plt.plot(range(repetitions), cost_array)
        plt.show()
        hs = [self.h_function(example) for example in range(self.num_examples)]
        ys = [self.target[i] for i in range(self.num_examples)]
        plt.plot(hs, ys, color = 'g', marker = 'o', linewidth = 0)
        plt.show()

    def scaling(self):
        m = np.zeros((self.num_examples, self.num_features + 1)) + 1
        for i in range(1, len(self.features[0])):
            average = sum(self.features[:, i])/self.num_examples
            maxx = max(self.features[:, i])
            m[:, i] = (self.features[:, i]-average)/maxx
        self.features = m

    def r2(self):
        pass
        enum_list = []
        denom_list = []
        for i in range(self.num_examples):
            enum_list.append((self.target[i]-self.h_function(i))**2)
            average = sum(self.target)/self.num_examples
            denom_list.append((self.target[i]-average)**2)
        enum = sum(enum_list)
        denom = sum(denom_list)
        return 1 - enum/denom


boston = datasets.load_boston()
features = boston.data
features_subset = np.vstack(features[:, 0])
target = boston.target

important_features = np.delete(boston.data, [2, 6], 1)

c = RegularizedLinearRegression(features, target)  # full boston dataset

c.scaling()
c.gradient_descent2(0.5, 0)
print(c.r2())
print(c.initial_parameters[0], np.mean(c.target))
