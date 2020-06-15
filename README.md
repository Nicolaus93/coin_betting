# optimal-pytorch

### Installation

You can easily install the library and all necessary dependencies by running `pip install -e .` from the root folder.


### Usage

To use any optimization method, simply import it, for example:

```
from optimal_pytorch import Adam, SGD, SGDOL
```

For running `examples/profiler.py` , line_profiler is needed, if it is giving error via `pip install line_profiler` or in setup.py, follow the following steps :

```
git clone https://github.com/rkern/line_profiler.git
find line_profiler -name '*.pyx' -exec cython {} \;
cd line_profiler && pip install . --user 
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