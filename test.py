import numpy as np

class LinearRegression(object):

    def __init__(self, features=np.array([]), target=np.array([]), initial_parameters=None, learning_rate=1):
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
        """
        returns the derivative of the cost function wrt the parameter with index parameter_index
        :param parameter_index: the index of the parameter
        :return: derivative with respect to the specified parameter of the cost function
        """
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



features = np.array([[1], [2], [3]])
target = np.array([1 ,2 ,3])

a = LinearRegression(features, target, [0,1])
print(a.features)
print(a.target)

print(a.initial_parameters)


print('cost:', a.cost())

a.gradient_descent(10)




