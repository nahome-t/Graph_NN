# MPhys project, investigating the inductive bias of graph neural network

On investingating the inductive bias of graph neural networks, the goal is to investigate two kinds of neural neworks, one refered to as GfNN as implemented in https://arxiv.org/pdf/1905.09550.pdf and the other just being a regular graph convolutional neural network. The paper investigates the removal of a lot of the linear layers within a graph neural network showing very little difference in the overall performance, furthermore inferring that the usefuleness in the adjacency layers is simply a de-noising effect.

From a graph signal processing perspective the research shows that repeatedly applying the normalised augmented adjacency layer (i.e. just performing the message passing) just causes the network to behave as an almost low pass filter for the eigenvalues of the normalised augmented laplacian matrix. This paired with the assumption that the underlying signal itself is purely just a low frequency signal with some high frequency noise gives a theoretical understanding.



# Structure of project
- Stage 1 initialise GCN layers randomly and produce a rank vs probability plot in order to see how often certain functions appear, do investigation for GNN of depths 2 and 10
-Stage 2 Train GCN such that it reaches 100% accuracy on trainning data then test it on testing data, then effectively redo

# Structure of program
- Model file contains the structure of both the regular neural neural network and the GfNN, also contains code defining the message passing layer
- Data Handler contains functions related to processing and recording the data, i.e. produces the mask which will show the dataset that we will test our neural networks on, will look at stored tensors and count the occurence of the 'functions' which are defined by their performance on the test set and can check as to whether some data is consistent with the data or the accuracy of a model
- Main just used if you want to run the neural network a single time and train it on the Cora dataset, will offer you choice in what type, configured so that it will be trained, can be used as a means of testing whether a particular GNN behaves well
- Runner is the code that will eventually be run the entire thing, for now will
