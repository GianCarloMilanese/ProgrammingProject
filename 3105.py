import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from scipy.stats import pearsonr
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
                for i in range(30, len(self.features[0])):
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
                for i in range(30, len(self.features[0])):
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

overxcrim = 1/CRIM
logdis= np.log(DIS)
lograd = np.log(RAD)
featureswlogdis = np.append(features, logdis, 1)
logtarget = np.log(target)
important_features = np.delete(features, [2, 6], 1)
impfeatwlogdis = np.append(important_features, logdis, 1)
featlogdislograd = np.append(featureswlogdis, lograd, 1)
withoverx = np.append(featlogdislograd, overxcrim, 1)
CRIMSCALED = (CRIM - np.mean(CRIM))/max(CRIM)
LSTATSCALED = (LSTAT - np.mean(LSTAT))/max(LSTAT)

def add_interaction(features, f1, f2):
    f1scaled = (f1 - np.mean(f1))/max(f1)
    f2scaled = (f2 - np.mean(f2)) / max(f2)
    p = f1scaled*f2scaled
    return np.append(features, p, 1)

f = featureswlogdis
f1 = add_interaction(f, AGE, NOX)
f2 = add_interaction(f1, RAD, TAX)
f3 = add_interaction(f2, INDUS, NOX)
#f4 = add_interaction(f3, NOX, DIS)
f4 = add_interaction(f3, RM, LSTAT)
f5 = add_interaction(f4, AGE, DIS)
f6 = add_interaction(f5, LSTAT, AGE)
f7 = add_interaction(f6, TAX, NOX)
f8 = add_interaction(f7, NOX, LSTAT)
f9 = add_interaction(f8, INDUS, LSTAT)
f10 = add_interaction(f9, INDUS, DIS)
f11 = add_interaction(f10, INDUS, AGE)
f12 = add_interaction(f11, ZN, DIS)
f13 = add_interaction(f12, TAX, INDUS)


c = LinearRegression(f13, logtarget)
d = LinearRegression(LSTAT, target)


c.scaling()
c.gradient_descent(1)
print(c.parameters[0], np.mean(c.target))  # test if theta0 = average
print(c.r2())
#print(c.parameters)



def plot_feature(feature, xlabel='feature_name', ylabel='PRICE'):
    plt.plot(feature, target, ".")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_sorted_feature_x(feature, xlabel='feature_name', ylabel='PRICE'):
    sort_pairs = sorted(zip(feature, target), key=lambda tup: tup[0])
    sort_bs, sort_cs = list(zip(*sort_pairs))
    plt.plot(sort_bs, sort_cs, ".")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_sorted_feature_y(feature, xlabel='feature_name', ylabel='PRICE'):
    sort_pairs = sorted(zip(feature, target), key=lambda tup: tup[1])
    sort_bs, sort_cs = list(zip(*sort_pairs))
    plt.plot(sort_bs, sort_cs, ".")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

FEATURE_NAMES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

CRIMSCALED = (CRIM - np.mean(CRIM))/max(CRIM)
LSTATSCALED = (LSTAT - np.mean(LSTAT))/max(LSTAT)

#plot_sorted_feature_x(np.log(CRIMSCALED*LSTATSCALED), xlabel="CRIM*LSTAT")

"""
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        if not i ==j:
            r, p_val = pearsonr(FEATURES[i], FEATURES[j])
            print(FEATURE_NAMES[i], FEATURE_NAMES[j], ": ", r, p_val)
            if r < -0.5 or r > 0.5:
                plt.plot(FEATURES[i], FEATURES[j], ".")
                plt.xlabel(FEATURE_NAMES[i])
                plt.ylabel(FEATURE_NAMES[j])
                plt.show()

plt.plot(LSTATSCALED, CRIMSCALED, ".")
plt.show()

for i in range(len(FEATURES)):
    plot_sorted_feature_x(FEATURES[i], FEATURE_NAMES[i])


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