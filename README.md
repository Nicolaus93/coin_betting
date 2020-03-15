# optimal-pytorch

### Installation

You can easily install the library and all necessary dependencies by running `pip install -e .` from the root folder.


### Usage

To use any optimization method, simply import it, for example:

```
from optimal_pytorch import Adam, SGD, SGDOL
```

### Example

We implemented an example on how to use the library to optimize a quadratic function which could be found in `quadratic.py` under  `tests`. Run `python ./tests/quadratic.py --help` for a detailed list and explanation of all running options. Example commands are:

```shell
python ./tests/quadratic.py --num-steps 50 --noise-std 0.1 --optim-method SGD --lr 0.01

python ./tests/quadratic.py --num-steps 50 --noise-std 0.1 --optim-method Adam --lr 0.1

python ./tests/quadratic.py --num-steps 50 --noise-std 0.1 --optim-method SGDOL --smoothness 100 --alpha 100
```
