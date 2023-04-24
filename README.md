# MPhys project, investigating the inductive bias of graph neural network

On investigating the inductive bias of graph neural networks, the goal is to investigate two kinds of neural neworks, one refered to as GfNN as implemented in https://arxiv.org/pdf/1905.09550.pdf and the other just being a regular graph convolutional neural network. The paper investigates the reordering of the adjacency layers within a graph neural network showing very little difference in the overall performance when placed at the start (almost as a smoothing filter on the dataset), furthermore inferring that the usefuleness in the adjacency layers is simply a de-noising effect.

From a graph signal processing perspective the research shows that repeatedly applying the normalised augmented adjacency layer (i.e. just performing the message passing) just causes the network to behave as a low pass filter for the eigenvalues of the normalised augmented laplacian matrix or in another words they function as mostly a smoothing or denoising effect. This paired with the assumption that the underlying signal itself is purely just a low frequency signal with some high frequency noise gives a theoretical understanding as to why the performance doesn't differ much.

The first part of the project involves looking at the frequency at which certain functions occur in both when the GfNN or GNN when it has weights which are randomly selected and when each model is trained up to some threshold accuracy (100% in the case of CiteSeer and 94% in the case of our synthetic dataset), the low dimensionality of the synthetic dataset makes it difficult to fit to 100% accuracy. Here a function is defind by how it performs in a generated dataset.

# Structure of project
- Stage 1: initialise GCN layers randomly and produce a rank vs probability plot in order to see how often certain functions appear, do investigation for GNN of depths 2 and 10 
- Stage 2: Train GCN such that it reaches 100% accuracy on trainning data then test it on testing data, then effectively redo
- Stage 3: Create a sythetic two circles dataset and repeat entire process for that to see how the performance varies


# Structure of program
- Model file contains the structure of both the regular neural neural network and the GfNN, also contains code defining the message passing layer
- Data Handler contains functions related to processing and recording the data, i.e. produces the mask which will show the dataset that we will test our neural networks on, will look at stored tensors and count the occurence of the 'functions' which are defined by their performance on the test set and can check as to whether some data is consistent with the data or the accuracy of a model. Also used to produce plots of rank v probability and perform changes to the data 
- Main just used if you want to run the neural network a single time and train it on the Cora dataset, will offer you choice in what type, configured so that it will be trained, can be used as a means of testing whether a particular GNN behaves well
- Runner is the code that will eventually be run on computer, will keep generating neural networks then looks at output on benchmark dataset, defined by benchmark_mask. Can currently run a test a specified amount of times and output the results into a file, can also download the file and look at how often each tensor or 'function' appears
- Analysis brings it all together and produces the rankVprob and probVprob plots very simply, automatically reads files via data_handler and puts it all togther to produce plots

# Stuff to-do
...
