import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class LinearRegression(object):

    def __init__(self, features=np.array([]), target=np.array([]), feat_names = None):
        """Constructor

        :param features: A matrix of features
        :param target: An array
        """

        # the number of features
        self.num_features = len(features[0])
        # the names of the features
        self.feat_names = feat_names if feat_names is not None else ['Feature {}'.format(i+1) for i in range(self.num_features)]
        # the number of examples
        self.num_examples = len(features)
        # an array of weights
        self.parameters = np.array([0 for i in range(self.num_features + 1)])
        # add a first column of 1's to the feature matrix
        m = np.zeros((self.num_examples, self.num_features+1))+1
        m[:, 1:] = features
        # feature matrix
        self.features = m
        # target array
        self.target = target

    def h_function(self):
        """ Hypothesis function

        :return: dot product of the parameters array and the feature matrix
        """
        return np.dot(self.parameters, np.transpose(self.features))

    def cost(self):
        """
        :return: Value of the cost function
        """
        return np.sum((self.h_function()-target)**2)

    def cost_der(self, parameter_index):
        """ Returns the derivative of the cost function wrt the parameter with index 'parameter_index'

        :param parameter_index: the index of the parameter
        :return: derivative with respect to the specified parameter of the cost function
        """
        e = (self.h_function()-self.target)*self.features[:, parameter_index]
        return (np.sum(e))/self.num_examples

    def update_parameters(self, learning_rate=1):
        """ Performs a simultaneous update of parameters.

        :param learning_rate: learning rate
        """
        new_parameter_list = []
        for i in range(len(self.parameters)):
            new_parameter_list.append(self.parameters[i] -
                                      learning_rate*self.cost_der(i))
        self.parameters = np.array(new_parameter_list)

    def plot_features_h(self, title='', ran=None ):
        """ Shows plots of features and of the hypothesis function wrt that feature

        :param title: title of the plot
        :param feat_names: the names of the features
        :param ran: the range of features that are plotted. By default it plots all the features
        """
        ran = ran if ran is not None else len(self.features[0])
        if ran > len(self.features[0]):
            ran = len(self.features[0])
        for i in range(1, ran):
            plt.plot(self.features[:, i], self.target, '.')
            theta0 = self.parameters[0]
            thetai = self.parameters[i]
            xs = np.linspace(min(self.features[:, i]), max(self.features[:, i]), 100)
            ys = [theta0 + thetai * x for x in xs]  # this is the h function
            plt.plot(xs, ys, color='r')
            plt.xlabel('{}'.format(self.feat_names[i-1]))
            plt.ylabel('PRICE')
            plt.title(title)
            plt.show()

    def scaling(self):
        """ Scales all the features so that they all are in a (-1, 1) range
        """
        m = np.zeros((self.num_examples, self.num_features + 1)) + 1
        for i in range(1, len(self.features[0])):
            average = sum(self.features[:, i])/self.num_examples
            maxx = max(self.features[:, i])
            m[:, i] = (self.features[:, i]-average)/maxx
        self.features = m

    def r2(self):
        """Calculates the coefficient of determination
        """
        enum = (self.target-self.h_function())**2
        denom = (self.target-np.mean(self.target))**2
        return 1 - np.sum(enum)/np.sum(denom)

    def gradient_descent(self, learning_rate, freq_plots=1000, ran=None):
        """ Performs the gradient descent algorithm.
        Every 'freq_plots' iterations and at convergence, plots features and hypothesis function
        After convergence shows the plot of the value of the cost function at each iteration;
                        of the actual prices vs the predicted prices
                        and prints the coefficient of determination.

        :param learning_rate: learning rate
        :param freq_plots: how often the features and the hypothesis function are plotted.
                            By default, every 1000 iterations
        :param ran: how many of the features are plotted every 'freq_plots' iterations.
                    By default: all of them
                    ran = 1 : None of them
                    e.g. ran = 5 : shows the plots of the first 4 features
        """
        self.scaling()
        repetitions = 0
        cost_list = []
        while True:
            cost_before = self.cost()
            cost_list.append(self.cost())
            self.update_parameters(learning_rate)
            if repetitions % freq_plots == 0:
                self.plot_features_h('Iteration {}'.format(repetitions),ran)
            repetitions += 1
            if np.abs(cost_before - self.cost()) < 0.0001:  # tests convergence
                self.plot_features_h('Convergence at iteration {}'.format(repetitions), ran)
                break
        cost_array = np.array(cost_list)
        plt.plot(range(repetitions), cost_array)
        plt.xlabel('REPETITIONS')
        plt.ylabel('COST')
        plt.show()
        hs = self.h_function()
        ys = self.target
        plt.plot(hs, ys, '.')
        plt.xlabel('PREDICTED PRICES')
        plt.ylabel('ACTUAL PRICES')
        plt.show()
        print(self.r2(), '\n')


if __name__ == "__main__":

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
    FEATURE_NAMES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    INTER_NAMES = ['AGE+NOX', 'RAD+TAX', 'INDUS+NOX', 'RM+LSTAT', 'AGE+DIS', 'LSTAT+AGE', 'TAX+NOX', 'NOX+LSTAT',
                   'INDUS+LSTAT', 'INDUS+DIS', 'INDUS+AGE', 'ZN+DIS', 'TAX+INDUS']

    def add_interaction(features, f1, f2):
        """

        :param features: a matrix of features
        :param f1: a feature array
        :param f2: a feature array
        :return: product of (scaled) f1 and f2 appended to features
        """
        f1scaled = (f1 - np.mean(f1))/max(f1)
        f2scaled = (f2 - np.mean(f2)) / max(f2)
        p = f1scaled*f2scaled
        return np.append(features, p, 1)

    # log of target
    logtarget = np.log(target)
    # lof of DIS
    logdis = np.log(DIS)
    # features + log of DIS
    featureswlogdis = np.append(features, logdis, 1)
    # the following adds to 'features + logdis' a number of columns corresponding to interactions between features
    f = featureswlogdis
    f1 = add_interaction(f, AGE, NOX)
    f2 = add_interaction(f1, RAD, TAX)
    f3 = add_interaction(f2, INDUS, NOX)
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

    # candidates for baseline
    BASELINE1 = LinearRegression(RM, target, ['RM'])                # r2 = 0.48
    BASELINE2 = LinearRegression(LSTAT, target, ['LSTAT'])             # r2 = 0.54
    BASELINE3 = LinearRegression(features, target, FEATURE_NAMES)          # r2 = 0.74

    # improvement #1
    LOGDIS = LinearRegression(featureswlogdis, logtarget, FEATURE_NAMES+['LOGDIS'])   # r2 = 0.8000014145388733

    # improvement #2
    INTERACTIONS = LinearRegression(f13, logtarget, FEATURE_NAMES+['LOGDIS']+INTER_NAMES)  # r2 = 0.8441889821747711

    # Run gradient descent for the implementations and prints the coefficient of determination
    # change the value of ran depending on how many plots you want to see.
    # e.g. ran= 1 --> no plot ; ran=None --> all plots ; ran=4 --> first 3 features
    print('Coefficient of determination for BASELINE1:')
    BASELINE1.gradient_descent(1.5, 600, ran=1)
    print('Coefficient of determination for BASELINE2:')
    BASELINE2.gradient_descent(1, 200, ran=1)
    print('Coefficient of determination for BASELINE3:')
    BASELINE3.gradient_descent(1, 900, ran=1)
    print('Coefficient of determination for LOGDIS:')
    LOGDIS.gradient_descent(1, 3000, ran=1)
    print('Coefficient of determination for INTERACTIONS:')
    INTERACTIONS.gradient_descent(1.2, 10000, ran=1)   # takes around 20000 iterations
