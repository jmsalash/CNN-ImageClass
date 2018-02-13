# CNN-ImageClass
Python project for the Machine Learning module of the MSc AI and Robotics in Sapienza University of Rome

This homework implements a convolutional neural network using Tensorflow to classify 3 types of boats from the dataset available under:
http://www.dis.uniroma1.it/~labrococo/MAR/classification.htm

It can be reused to to classify any other images just by changing the following variables at the top of the code:
classes: an array with the class names, which should be the same as the folder names.
train_path: where the image folders to classify are located.

To do this it has been used a configuration of a 5x5 kernel and a 2x2 pool subsampling with the following parameters reconfigured during the experiments:
- Number of convolutional layers (between 2 and 4).
- Number of starting features.
- Number of output feature reduction.
- Keep probability of the dropout node.

A dropout node was added to the graph, applied to the 1st fully connected layer. This was added to avoid over-fitting.
ReLU was used as the non-linear function for all the neural network layers.

An Adam optimiser was used to minimise the loss function and train the neural network, with a learning rate of 1e-4. 



