**Description**
This repository contains the code to build and test the KKTNet model.

The KKTNet is a neural network architecture used to solve convex optimization problems. This network integrates the Karush-Kuhn-Tucker(KKT) conditions, which are necessary and sufficient for the solution of convex optimization problems, into the loss functions of the network. 

The gen_dataset.py file contains the code to generate linear programs with the coefficients normalized. This dataset is used to train the KKTNet model. We use cvxpy to generate and solve the problems.

The kktnet_model.py file contains the code to build the KKTNet model and train it on the dataset generating using the gen_dataset.py file. 
