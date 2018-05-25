import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):

    def __init__(self, features=np.array([]), target=np.array([]), initial_parameters=None, learning_rate=0.1): #maybe no need for empty array
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
        for i in range(repetitions):
            print('before:', self.initial_parameters)
            self.update_parameters()
            print('after:', self.initial_parameters)



features = np.array([[1], [2], [3]])
target = np.array([1 ,2 ,3])

a = LinearRegression(features, target, [0.2,0.3])
print(a.features)
print(a.target)

print(a.initial_parameters)


print('cost:', a.cost())
#print in terms of rounds?
a.gradient_descent(5000)

"""plt.semilogy(range(10), a.cost)
plt.title("L")
plt.show()"""


#Define a graph - iterations, cost function (to see that it is decreasing)
