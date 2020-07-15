# optimal-pytorch

### Installation

You can easily install the library and all necessary dependencies by running `pip install -e .` from the root folder.


### Usage

To use any optimization method, simply import it, for example:

```
from optimal_pytorch.optim import Adam, SGD, SGDOL
```

To use the functions we have designed as the unit tests, you can access them by running : 
```
from optimal_pytorch.functions.loss_functions_1d import AbsoluteLoss, GaussianLoss
```

### Example

We implemented an example on how to use the library to optimize a quadratic function which could be found in `quadratic.py` under  `tests`. Run `python ./tests/quadratic.py --help` for a detailed list and explanation of all running options. Example commands are:

```shell
python ./tests/quadratic.py --num-steps 50 --noise-std 0.1 --optim-method SGD --lr 0.01

python ./tests/quadratic.py --num-steps 50 --noise-std 0.1 --optim-method Adam --lr 0.1

python ./tests/quadratic.py --num-steps 50 --noise-std 0.1 --optim-method SGDOL --smoothness 100 --alpha 100
```

Another example implements a simple neural network on MNIST dataset in examples/mnist.py, To run that just go to examples folder and run:

```
python mnist.py --optimizer optimizer_name
python mnist.py --opt optimizer_name
```
### Running Unit tests

We have implemented a bunch of simple functions on which these optimizers are run on for a while. The file test_optimizers_mp.py is present in Experimental folder, it can be run by :
```
python test_optimizers_mp.py --plot
```

to plot the results and by running : 
```
python test_optimizers_mp.py --suboptimal
```
to check for suboptimality gap (default L1 norm).


In addition, currently, there is a script plot_results.py which will look for the result files and compile them by:
```
python plot_results.py --plot --metric suboptimal/l1

python plot_results.py --interpret --metric suboptimal/l1
```
to plot the results or interpret them.

### Possible Issues

For running `examples/profiler.py` , line_profiler is needed, if it is giving error via `pip install line_profiler` or in setup.py, follow the following steps :

```
git clone https://github.com/rkern/line_profiler.git
find line_profiler -name '*.pyx' -exec cython {} \;
cd line_profiler && pip install . --user 
```