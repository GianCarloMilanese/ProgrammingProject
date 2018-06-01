The following are the contents of this folder.

Report.pdf : a report with our findings.

LinearRegression.py : an implementation of linear regression in Python.
          The code contains a class named LinearRegression.
          Executing the code will run the gradient descent algorithm for 5 models for the Boston Housing dataset.
              The MEDVvsRM model: takes account of the RM feature only
              The MEDVvsLSTAT model: takes account of the LSTAT feature only
              The MEDVvsALL model: takes account of all the features in the Boston Housing dataset
              The LOGDIS model:  also takes account of the logarithms of the DIS feature
              The INTER model: also takes account of some interactions between features
          The value of the coefficient of determination is printed for each model.
          Some plots are also showed (more details in the code).
          The following libraries are needed to run the code: numpy, matplotlib, scikit-learn.

The original dataset can be found here: http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html