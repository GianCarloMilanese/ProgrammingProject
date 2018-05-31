import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import warnings
import math


class LinearRegression(object):

    def __init__(self, features=np.array([]), target=np.array([])): #maybe no need for empty array
        self.num_features = len(features[0]) #len of first row
        self.num_examples = len(features) # = 506
        self.parameters = np.array([1 for i in range(self.num_features + 1)]) # array of 1's
        m = np.zeros((self.num_examples, self.num_features+1))+1 #matrix of all 1s: 506*14
        m[:, 1:] = features
        self.features = m #not number of features, just a placeholder
        self.target = target


    "The hypothesis function = theta^t * x"
    def h_function(self):
        return np.dot(self.parameters, np.transpose(self.features))

    "The Cost function J(theta)=1/2m*sum_{i=1^m}(h_theta(x^i)-y^i)^2"

    def cost(self):
        return np.sum((self.h_function()-target)**2)

    def cost_der(self, parameter_index):
        #not sure if parameter_index needs to be in there?
        """
        returns the derivative of the cost function wrt the parameter with index parameter_index
        :param parameter_index: the index of the parameter
        :return: derivative with respect to the specified parameter of the cost function
        """
        e = (self.h_function()-self.target)*self.features[:, parameter_index]
        return (np.sum(e))/self.num_examples

    def update_parameters(self, learning_rate=0.01):
        new_parameter_list = [] #simultaneous update of parameters
        for i in range(len(self.parameters)):
            new_parameter_list.append(self.parameters[i] -
                                      learning_rate*self.cost_der(i))
        self.parameters = np.array(new_parameter_list)

    def gradient_descent2(self, learning_rate):
        repetitions = 0
        cost_list = []
        while True:
            cost_before = self.cost()
            cost_list.append(self.cost())
            self.update_parameters(learning_rate)

            if repetitions%20000 == 0:   # this plots the hypothesis function; works for the basic case with one feature
                for i in range(1, len(self.features[0])):
                    plt.plot([feature[i] for feature in self.features], self.target, color='g', linewidth=0, marker='o')
                    theta0 = self.parameters[0]
                    theta1 = self.parameters[i]
                    xs = np.linspace(min(self.features[:, i]), max(self.features[:, i]), 100)
                    ys = [theta0 + theta1 * x for x in xs]  # this is the h function
                    plt.plot(xs, ys, color='r')
                    plt.title('Iteration {}, feature number {}'.format(repetitions, i))
                    plt.show()
            repetitions += 1
            if np.abs(cost_before - self.cost()) < 0.0001:  # tests convergence
                for i in range(1, len(self.features[0])):
                    plt.plot([feature[i] for feature in self.features], self.target, color='g', linewidth=0, marker='o')
                    theta0 = self.parameters[0]
                    theta1 = self.parameters[i]
                    xs = np.linspace(min(self.features[:, i]), max(self.features[:, i]), 100)
                    ys = [theta0 + theta1 * x for x in xs]  # this is the h function
                    plt.plot(xs, ys, color='r')
                    plt.title('Convergence at iteration {}, feature number {}'.format(repetitions, i))
                    plt.show()
                break
        cost_array = np.array(cost_list)
        plt.plot(range(repetitions), cost_array)
        plt.show()
        hs = self.h_function()
        ys = self.target
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
        enum = (self.target-self.h_function())**2
        average = np.sum(self.target) / self.num_examples
        denom = (self.target-average)**2
        return 1 - np.sum(enum)/np.sum(denom)


boston = datasets.load_boston()
features = boston.data
features_subset = np.vstack(features[:, 0])
target = boston.target
RN = np.vstack(features[:, 5])
AGE = np.vstack(features[:, 6])
distances = np.vstack(features[:, 7])

logdis= np.sqrt(distances)
featureswlogdis = np.append(features, logdis, 1)

logtarget = np.log(target)

c = LinearRegression(featureswlogdis, logtarget)  # full boston dataset
#print(logtarget)
c.scaling()
c.gradient_descent2(1)
print(c.parameters[0], sum(c.target)/c.num_examples)
#print(c.target)
#print(c.h_function())
print(c.r2())




#print(c.initial_parameters[0], np.mean(c.target))


"""
lvls = [len(set(v)) for v in features.T]
names = boston["feature_names"]

print([(l, n) for l, n in zip(lvls, names)])



cs = features[:, 0]
bs = features[:, 11]

sort_pairs = sorted(zip(bs, cs), key=lambda tup: tup[0])

sort_bs, sort_cs = list(zip(*sort_pairs))

plt.plot(sort_bs, sort_cs, ".")

plt.ylabel("Prop of crime")
plt.xlabel("Prop of blacks")
plt.legend()
plt.title("Correlation")
plt.show()

plt.plot(range(len(target)), target, linewidth = 0, marker = 'o')
plt.show()
"""