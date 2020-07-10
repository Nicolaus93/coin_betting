class MNIST_Config():
    def __init__(self,
                 batch_size=60,
                 test_batch_size=1000,
                 lr=1e-3,
                 epochs=10):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.epochs = epochs

class Unit_Test_Config():
    def __init__(self,
                 num_iterations=1e4,
                 num_runs=5,
                 num_func_variations = 5):
        self.num_iterations = num_iterations
        self.num_runs = num_runs
        self.num_func_variations = num_func_variations