import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class LinearRegression(object):

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
    def cost(self):
        my_list = []
        for i in range(self.num_examples):
            my_list.append((self.h_function(i)-self.target[i])**2)
        s = sum(my_list)
        return (1/(2*self.num_examples))*s

    def cost_der(self, parameter_index):
        #not sure if parameter_index needs to be in there?
        """
        returns the derivative of the cost function wrt the parameter with index parameter_index
        :param parameter_index: the index of the parameter
        :return: derivative with respect to the specified parameter of the cost function
        """
        my_list = []
        for i in range(self.num_examples):
            my_list.append((self.h_function(i)-self.target[i])*self.features[i][parameter_index])
        return sum(my_list)/self.num_examples

    def update_parameters(self, learning_rate=0.01):
        new_parameter_list = [] #simultaneous update of parameters
        for i in range(len(self.initial_parameters)):
            new_parameter_list.append(self.initial_parameters[i] -
                                      learning_rate*self.cost_der(i))
        self.initial_parameters = np.array(new_parameter_list)

    def gradient_descent(self, repetitions, learning_rate):
        cost_list = []
        for i in range(repetitions):
            c = self.cost()
            cost_list.append(self.cost())
            self.update_parameters(learning_rate)
            if i%(repetitions/5) == 0:
                plt.plot([feature[1] for feature in self.features], self.target, color='g', linewidth=0, marker='o')
                xs = np.linspace(0, max(np.hstack(self.features)), 100)
                theta0 = self.initial_parameters[0]
                theta1 = self.initial_parameters[1]
                ys = [theta0 + theta1*x for x in xs] #this is the h function
                plt.plot(xs, ys, color='r')
                plt.show()
            if c - self.cost() < 0.0001:
                print('convergence at iteration', i)
        cost_array=np.array(cost_list)
        plt.plot(range(repetitions),cost_array)
        plt.show()

    def gradient_descent2(self, learning_rate):
        repetitions = 0
        cost_list = []
        while True:
            cost_before = self.cost()
            cost_list.append(self.cost())
            self.update_parameters(learning_rate)
            """
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
            """
            repetitions += 1
            if cost_before - self.cost() < 0.0001: # tests convergence
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

    def scaling(self):
        m = np.zeros((self.num_examples, self.num_features + 1)) + 1
        for i in range(1, len(self.features[0])):
            average = sum(self.features[:, i])/self.num_examples
            maxx = max(self.features[:, i])
            m[:, i] = (self.features[:, i]-average)/maxx
        self.features = m

    def r2(self):





boston = datasets.load_boston()
features = boston.data
features_subset = np.vstack(features[:, 0])
target = boston.target

a = LinearRegression(features_subset, target)

matrix = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
t = np.array([20, 18, 22, 15, 13, 17, 10, 12, 9, 4, 3])

a = LinearRegression(features_subset, target)  # Only CRIM feature
b = LinearRegression(matrix, t)  # small example with one variable
c = LinearRegression(features, target)  # full boston dataset


#b.gradient_descent2(0.03)
#b.scaling()
#b.gradient_descent2(0.1)


#print(c.features[:,1])

features_without_chas = np.delete(boston.data, 3, 1)  # deletes column with index 3

print(a.features)
print(min(a.features[:,1]))
print(max(a.features[:,1]))
#a.scaling()
print(min(a.features[:,1]))
print(max(a.features[:,1]))
print(a.features)

d = LinearRegression(features_without_chas, target)

#a.gradient_descent2(0.01)
d.scaling()
d.gradient_descent2(0.5)