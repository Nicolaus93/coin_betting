## Description
1. contains different data configurations that are used in different scripts throughout the program.
2. func_constraints contains constraints for every function, any new loss function added needs to be added here to be run in test_optimizers script.
3. opt_params contains different hyperparameters for each optimizer to run on. any new optimizers function added needs to be added here to run in test_optimizers_script.
4. MNIST data/Any external dataset is kept here.
5. config.py contains different config objects for different scripts (examples/mnist.py, experimental/test_optimizers_mp.py).