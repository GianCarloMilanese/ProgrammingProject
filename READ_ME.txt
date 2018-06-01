The following are the contents of this folder.

report.pdf : a report with our findings.

code.py : an implementation of linear regression in Python.
          The code contains a class named LinearRegression.
          Executing the code will run the gradient descent algorithm for 5 models for the Boston Housing dataset.
            The BASELINE1 model: takes account of the RM feature only
            The BASELINE2 model: takes account of the LSTAT feature only
            The BASELINE3 model: takes account of all the features in the Boston Housing dataset
            The LOGDIS model:  also takes account of the logarithms of the DIS feature
            The INTERACTION model: also takes account of some interactions between features
          The value of the coefficient of determination is printed for each model.
          Some plots are also showed (more details in the code).

The original dataset can be found here: http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html