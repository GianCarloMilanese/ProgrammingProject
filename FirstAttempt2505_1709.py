import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

boston = datasets.load_boston()
features = boston.data
#features = np.vstack(features)
target = boston.target

class LinearRegression(object):

    def __init__(self, features=np.array([]), target=np.array([]), initial_parameters=None, learning_rate=0.5): #maybe no need for empty array
        self.num_features = len(features[0]) #len of first row
        self.num_examples = len(features) # = 506
        self.initial_parameters = np.array(initial_parameters) #if initial parameters = undefined, we need to randomly assign?
        m = np.zeros((self.num_examples, self.num_features+1))+1 #matrix of all 1s: 506*14
        m[:, 1:] = features
        self.features = m #not number of features, just a placeholder
        self.target = target
        self.learning_rate = learning_rate

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

    def update_parameters(self):
        new_parameter_list = [] #simultaneous update of parameters
        for i in range(len(self.initial_parameters)):
            new_parameter_list.append(self.initial_parameters[i] -
                                      self.learning_rate*self.cost_der(i))
        self.initial_parameters = np.array(new_parameter_list)

    def gradient_descent(self, repetitions):
        cost_list=[]
        for i in range(repetitions):
            #print('before:', self.initial_parameters)
            cost_list.append(self.cost())
            print('cost', self.cost())
            self.update_parameters()
            print('after:', self.initial_parameters)
            if i%10 != 0:
                plt.plot([feature[1] for feature in self.features], self.target, color='g', linewidth = 0, marker='o')
                xs = np.linspace(0, 3, 10)
                theta0 = self.initial_parameters[0]
                theta1 = self.initial_parameters[1]
                ys = [theta0 + theta1*x for x in xs]
                plt.plot(xs, ys, color='r')
                plt.show()
        cost_array=np.array(cost_list)
        #print(cost_array)
        plt.plot(range(repetitions),cost_array)
        plt.axis([0, repetitions, 0, 4e+22])
        plt.show()


a = LinearRegression(features, target, [-100, 11987652, 123123100, 131300,  1012345678900, -34561, 13456780,
                                        -14345600, -234100, 100, -1040,  198765400, -12345622, 234562])
print(a.features)
print(a.target)

print(a.initial_parameters)


print('cost:', a.cost())
#print in terms of rounds?
#a.gradient_descent(100)

"""plt.semilogy(range(10), a.cost)
plt.title("L")
plt.show()"""


#Define a graph - iterations, cost function (to see that it is decreasing)

matrix = np.array([[0], [1], [2]])
t = np.array([0,1,2])

b = LinearRegression(matrix, t, [1,2])

#b.gradient_descent(100)
print(b.initial_parameters)
b.update_parameters()
print(b.initial_parameters)

b.gradient_descent(100)