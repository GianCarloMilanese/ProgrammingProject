import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class LinearRegression(object):

    def __init__(self, features=np.array([]), target=np.array([])): #maybe no need for empty array
        self.num_features = len(features[0]) #len of first row
        self.num_examples = len(features) # = 506
        self.parameters = np.array([0 for i in range(self.num_features + 1)]) # array of 1's
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
        """
        returns the derivative of the cost function wrt the parameter with index parameter_index
        :param parameter_index: the index of the parameter
        :return: derivative with respect to the specified parameter of the cost function
        """
        e = (self.h_function()-self.target)*self.features[:, parameter_index]
        return (np.sum(e))/self.num_examples

    def update_parameters(self, learning_rate=0.01):
        new_parameter_list = []  #simultaneous update of parameters
        for i in range(len(self.parameters)):
            new_parameter_list.append(self.parameters[i] -
                                      learning_rate*self.cost_der(i))
        self.parameters = np.array(new_parameter_list)

    def gradient_descent(self, learning_rate):
        repetitions = 0
        cost_list = []
        while True:
            cost_before = self.cost()
            cost_list.append(self.cost())
            self.update_parameters(learning_rate)

            if repetitions % 5000 == 0 and repetitions != 0:   # this plots the hypothesis function; works for the basic case with one feature
                for i in range(15, len(self.features[0])):
                    plt.plot([feature[i] for feature in self.features], self.target, color='g', linewidth=0, marker='o')
                    theta0 = self.parameters[0]
                    theta1 = self.parameters[i]
                    xs = np.linspace(min(self.features[:, i]), max(self.features[:, i]), 100)
                    ys = [theta0 + theta1 * x for x in xs]  # this is the h function
                    plt.plot(xs, ys, color='r')
                    plt.title('Iteration {}'.format(repetitions))
                    plt.xlabel('Feature {}'.format(i))
                    plt.ylabel('PRICE')
                    plt.show()
            repetitions += 1
            if np.abs(cost_before - self.cost()) < 0.0001:  # tests convergence
                for i in range(15, len(self.features[0])):
                    plt.plot([feature[i] for feature in self.features], self.target, color='g', linewidth=0, marker='o')
                    theta0 = self.parameters[0]
                    theta1 = self.parameters[i]
                    xs = np.linspace(min(self.features[:, i]), max(self.features[:, i]), 100)
                    ys = [theta0 + theta1 * x for x in xs]  # this is the h function
                    plt.plot(xs, ys, color='r')
                    plt.title('Convergence at iteration {}'.format(repetitions))
                    plt.xlabel('Feature {}'.format(i))
                    plt.ylabel('PRICE')
                    plt.show()
                break
        cost_array = np.array(cost_list)
        plt.plot(range(repetitions), cost_array)
        #plt.show()
        hs = self.h_function()
        ys = self.target
        plt.plot(hs, ys, color = 'g', marker = 'o', linewidth = 0)
        #plt.show()

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
target = boston.target

CRIM = np.vstack(features[:, 0])
ZN = np.vstack(features[:, 1])
INDUS = np.vstack(features[:, 2])
CHAS = np.vstack(features[:, 3])
NOX = np.vstack(features[:, 4])
RM = np.vstack(features[:, 5])
AGE = np.vstack(features[:, 6])
DIS = np.vstack(features[:, 7])
RAD = np.vstack(features[:, 8])
TAX = np.vstack(features[:, 9])
PTRATIO = np.vstack(features[:, 10])
B = np.vstack(features[:, 11])
LSTAT = np.vstack(features[:, 12])

FEATURES = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]

logdis= np.log(DIS)
lograd = np.log(RAD)
featureswlogdis = np.append(features, logdis, 1)
logtarget = np.log(target)
important_features = np.delete(features, [2, 6], 1)
impfeatwlogdis = np.append(important_features, logdis, 1)
featlogdislograd = np.append(featureswlogdis, lograd, 1)



c = LinearRegression(featlogdislograd, logtarget)
d = LinearRegression(LSTAT, target)

c.scaling()
c.gradient_descent(1)
print(c.parameters[0], np.mean(c.target))  # test if theta0 = average
print(c.r2())

def plot_feature(feature, xlabel='feature_name', ylabel='PRICE'):
    plt.plot(feature, target, ".")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_sorted_feature(feature, xlabel='feature_name', ylabel='PRICE'):
    sort_pairs = sorted(zip(feature, target), key=lambda tup: tup[0])
    sort_bs, sort_cs = list(zip(*sort_pairs))
    plt.plot(sort_bs, sort_cs, ".")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

FEATURE_NAMES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

"""
for i in range(len(FEATURES)):
    #print(zip(FEATURES[i], target) == sorted(zip(FEATURES[i], target), key=lambda tup: tup[0]))
    #plot_feature(FEATURES[i], FEATURE_NAMES[i])
    pass
    plot_sorted_feature(FEATURES[i], FEATURE_NAMES[i])
"""



"""
plot_feature(CRIM, 'CRIM')
plot_feature(logdis, 'LOGDIS')
plot_feature(RM, 'RM')
"""

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