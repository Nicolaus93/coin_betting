import numpy as np
import torch
import optimal_pytorch
import json
import os
import time
from collections import Counter
import multiprocessing
from itertools import repeat
from matplotlib import pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'tests')))
import functions
# For reproducibility.
torch.manual_seed(1)

# List of all exhaustive params of optimizers.
opt_params = {
    'Adam': {
        'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        'betas': [(0.9, 0.999), (0, 0.99)]
    },
    'SGD': {
        'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        'momentum': [0.1, 0.5, 0.9, 0.999],
        'nesterov': [False, True]
    },
    'SGDOL': {
        'smoothness': [10, 20],
        'alpha': [10, 20]
    }
}

# List all function parameters to use.
func_params = ['fprime', 'fs', 'xs', 'xe', 'mu', 'sd']
func_constraints = {
    'loss_x_sinx': {
        'xs': 0,
        'xe': 10
    },
    'loss_synthetic_func': {
        'xs': 0,
        'xe': 1
    },
    'loss_gaussian': {
        'xs': 1,
        'xe': 2
    }
}

# Color codes for plotting graphs.
color_rgb = {
    'r': [194, 24, 7],
    'g': [41, 171, 135],
    'b': [15, 82, 186],
    'y': [255, 255, 0],
    'o': [255, 170, 29],
    'p': [254, 127, 156]
}
# Number of iterations to run one function-optimizer combination on.
iterations = 100


# Generates a list of all possible hyperparameter combinations for optimizer as defined in params.
def generate_optimizers(params):
    opt_combinations = {}
    counter = 0
    for key in params:
        # Create an empty array if that key is empty [will happen in case of first key(here Adam)].
        if not key in opt_combinations:
            opt_combinations[key] = []
        # Traverse throough all keys(hyperparameters of optimizers) of params list.
        for prop in params[key]:
            if (len(opt_combinations[key]) == 0):
                # if the array is empty [will happen in case of first property(here lr)].
                for ele in params[key][prop]:
                    dic = {}
                    dic[prop] = ele
                    opt_combinations[key].append(dic)
            else:
                # Create n copies of the original list
                len_before = len(opt_combinations[key])
                temp = []
                for _ in range(len(params[key][prop])):
                    for j in opt_combinations[key]:
                        temp.append(j.copy())
                # Adding multiple copies to original object.
                opt_combinations[key] = temp
                # Populating the multiple copies with different values of other properties.
                for i in range(len(params[key][prop])):
                    for j in range(len_before):
                        opt_combinations[key][
                            j + i * len_before][prop] = params[key][prop][i]
        counter += len(opt_combinations[key])

    return opt_combinations, counter


# Generates random parameter combination for functions to be used in the experiments.
# Functions_list = A list of all the functions, func_params = a list of all the parameters every function requires(ex: xs, xe, mu)
# func_constraints = Functions which have constraints for the input space(have a specific minima in that domain).
def generate_functions(functions_list, func_params, func_constraints):
    scaling = 10
    dic = {}
    vars = torch.rand(len(func_params), dtype=torch.float) * scaling
    for name in functions_list:
        dic[name] = {}
        for i in range(len(func_params)):
            # If func_constraints contains constraints for some functions
            if (name in func_constraints):
                xs = func_constraints[name]['xs']
                xe = func_constraints[name]['xe']
                # Every other parameters except xs and xe are randomly generated within the range [xs, xe]
                if (not (func_params[i] == 'xs' or func_params[i] == 'xe')):
                    if (vars[i] > xe or vars[i] < xs):
                        dic[name][func_params[i]] = torch.rand(
                            1, dtype=torch.float) * (xe - xs) + xs
                    else:
                        dic[name][func_params[i]] = vars[i]
                # xs and xe values should stay the same and not be replaced by some random values
                elif (func_params[i] == 'xs'):
                    dic[name][func_params[i]] = torch.tensor(xs)
                elif (func_params[i] == 'xe'):
                    dic[name][func_params[i]] = torch.tensor(xe)
            # If that function has no constraints, we let the random values be
            else:
                dic[name][func_params[i]] = vars[i]
        #storing xs and xe so that we generate our initial x between these values
        if (not name in func_constraints):
            func_constraints[name] = {}
            func_constraints[name]['xs'] = np.array(
                vars[2].data.numpy()).reshape(1)[0]
            func_constraints[name]['xe'] = np.array(
                vars[3].data.numpy()).reshape(1)[0]
    return dic


# This function generates the initial value for every function-optimizer combination
# for some num_runs between [xs, xe] for every function.
def generate_initial(optimizer_name, functions_combinations, num_runs):
    init_tensors = []
    for i in range(num_runs):
        init_run = []
        for key in functions_combinations:
            xs = functions_combinations[key]['xs']
            xe = functions_combinations[key]['xe']
            combo = [
                torch.rand(1, dtype=torch.float) * (xe - xs) + xs
                for _ in range(len(optimizer_name))
            ]
            for ele in combo:
                ele.requires_grad_()
            init_run.append(combo)
        init_tensors.append(init_run)
    return init_tensors


# runs an optimizer instance on a function for n iterations
# func_name = which function is used, opt_name = which optimizer is used, func_params = the parameters of that specific function
# opt_params = hyper parameters for that specific optimizer, initial = the initial value to optimize.
def run_function(func_name, opt_name, func_params, opt_params, initial):
    global iterations
    loss_function = getattr(functions, func_name)(func_params[func_name])
    x = initial
    a = x.clone().data.numpy()[0]
    # Runs loss_function on optimizer for some iterations
    optimizer = getattr(optimal_pytorch, opt_name)([x], **opt_params)
    for i in range(iterations):
        optimizer.zero_grad()
        loss = loss_function.forward(x)
        loss.backward()
        optimizer.step()
    minima = loss_function.get_minima()
    if (torch.is_tensor(minima)):
        b = str(minima.data.numpy().reshape(1)[0])
    else:
        b = str(minima)
    c = str(x.data.numpy().reshape(1)[0])
    return a, b, c


"""Color coding used to plot the graphs
    red = if diff between reached and optimal solution is more than 0.5
    blue = if diff between reached and optimal solution is between 0.1 and 0.5
    green = if diff between reached and optimal solution is less than 0.1

    red + blue => pink, when red and blue are obtained in equal proportions in n runs
    red + green => yellow, when red and green are obtained in equal proportions in n runs
    green + blue => orange, when green and blue are obtained in equal proportions in n runs

    in all other cases the color which occurs max number of times is plotted
"""


# Function to assign colors based on the difference between values.
def compare_results(optimal, obtained):
    a = float(optimal)
    b = float(obtained)
    c = abs(a - b)
    if (c <= 0.01):
        return 'g'
    elif (c >= 0.01 and c <= 0.05):
        return 'b'
    else:
        return 'r'


# for cases where the max colors are same.
def mix_color(counter):
    if (counter[0][1] == counter[1][1]):
        temp = counter[0][0] + counter[1][0]
        if (temp == 'rg' or temp == 'gr'):
            return 'y'
        elif (temp == 'gb' or temp == 'bg'):
            return 'o'
        elif (temp == 'rb' or temp == 'br'):
            return 'p'
    else:
        return counter[0][0]


# Function which adds solutions generated during the optimization process to a json object.
# results = The json object which will contains all the results, func_name = the name of the function,
# func_params = the parameters of that function, soln contains the final values obtained in the optimization
#  process for an optimizer on a function, opt_names = list of all optimizers, opt_config = config for every
# optimizer in opt_names
def add_soln_to_results(results, func_name, func_params, soln, opt_names,
                        opt_config):
    for i in range(len(opt_names)):
        opt_name = opt_names[i]
        opt_conf = opt_config[i]
        str1 = ''
        for key in opt_conf:
            str1 += (key + '|' + str(opt_conf[key]) + '|')
        str2 = ''
        str2 += (func_name + '|')
        for key in func_params:
            str2 += (key + '|' +
                     str(func_params[key].data.numpy().reshape(1)[0]) + '|')
        results[opt_name][str1][str2]['x_initial'].append(str(soln[i][0]))
        results[opt_name][str1][str2]['x_optimal'].append(str(soln[i][1]))
        results[opt_name][str1][str2]['x_soln'].append(str(soln[i][2]))
        results[opt_name][str1][str2]['color'].append(
            str(compare_results(soln[i][1], soln[i][2])))


# iterator like function which gives the next optimizer and its hyperparameters.
def give_next_opt(opt_combinations, counter):
    temp = counter
    for opt_key in opt_combinations:
        if (temp >= len(opt_combinations[opt_key])):
            temp = temp - len(opt_combinations[opt_key])
            continue
        else:
            return opt_key, opt_combinations[opt_key][temp]


"""
Initializes results object with every entry, The results is finally in the format :
 { Opt_name1: { Opt_parameters1:{ function1:{ x_initial : [...],
                                             x_soln : [...],
                                             x_optimal: [...],
                                             color: [...]
                                            },
                                function2: {...}
                               },
                Opt_parameters2:{...}
               },
    Opt_name2:{...}
 }
"""


def add_result_to_json(results, opt_name, opt_params, functions_list,
                       func_params):

    for i in range(len(opt_params)):
        # Check if optimizer is already present in dict object
        if (not opt_name[i] in results):
            results[opt_name[i]] = {}
        # create an entry for a specific set of hyperparameters for the optimizer
        str1 = ''
        for key in opt_params[i]:
            str1 += (key + '|' + str(opt_params[i][key]) + '|')
        results[opt_name[i]][str1] = {}
        # Create an entry for the specific parameters for the function (will be same, since created using the same seed)
        for func_name in functions_list:
            str2 = ''
            str2 += (func_name + '|')
            for param in func_params[func_name]:
                if (not (param == 'min' or param == 'soln'
                         or param == 'initial')):
                    temp = func_params[func_name][param].data.numpy().reshape(
                        1)
                    str2 += (param + '|' + str(temp[0]) + '|')
            #creating entries for our starting point optimal solution and solution after 500 steps
            if (not str2 in results[opt_name[i]][str1]):
                results[opt_name[i]][str1][str2] = {}
                results[opt_name[i]][str1][str2]['x_optimal'] = []
                results[opt_name[i]][str1][str2]['x_soln'] = []
                results[opt_name[i]][str1][str2]['x_initial'] = []
                results[opt_name[i]][str1][str2]['color'] = []


# Helper function returning an array containing count of each entry.
def most_frequent(List):
    return Counter(List)


# Function to Plots results which were saved in results/opt_results.json file
def plot_results():
    global color_rgb, iterations
    with open('results/opt_results_' + str(iterations) + '.json') as file:
        results = json.load(file)
    compiled = {}
    for opt in results:
        # Selecting the most common occuring color of the n rounds
        compiled[opt] = {}
        for config in results[opt]:
            compiled[opt][config] = {}
            for functions in results[opt][config]:
                counter = most_frequent(
                    results[opt][config][functions]['color']).most_common(2)
                if (len(counter) > 1):
                    compiled[opt][config][functions] = mix_color(counter)
                else:
                    compiled[opt][config][functions] = counter[0][0]
    """Creates an n X m array where n = number of functions and m = optimizer 
    combinations and array values contain the data in the object 'compiled', 
    Finally color_arr contains color values like : 'r', 'g', 'b' etc.
    and rgb_arr contains a conversion of these colors into their rgb values as per color_rgb object.
    """
    # For every optimizer ex - Adam, SGD etc.
    for opt in compiled:
        optimizers_list = []
        functions_list = []
        title = opt
        color_arr = []
        rgb_arr = []
        # For every optimizer combination ex - lr : 0.1, momentum = 0.3 etc.
        for config in compiled[opt]:
            col_temp = []
            config_temp = config.split('|')
            str1 = ''
            for i in range(len(config_temp[:-1])):
                if (i % 2 == 0):
                    str1 += (config_temp[i][:2] + '|')
                else:
                    str1 += (config_temp[i] + '|')
            optimizers_list.append(str1)
            #For every function ex - loss_gaussian, loss_absolute
            for functions in compiled[opt][config]:
                pos = functions.find('|')
                color_val = compiled[opt][config][functions]
                # Create only one copy of function name in functions_list
                if (not functions[:pos] in functions_list):
                    functions_list.append(functions[:pos])
                col_temp.append(color_val)
            color_arr.append(col_temp)
        # reshaping it to n X m, where n = number of functions, m = number of optimizers
        color_arr = np.transpose(np.array(color_arr)).reshape(
            len(functions_list), len(optimizers_list))
        rgb_arr = np.array([[color_rgb[ele] for ele in row]
                            for row in color_arr])
        # plotting on heatmap and saving
        fig, ax = plt.subplots(figsize=(15, 40))
        plt.imshow(rgb_arr)
        # Ticks are just number of boxes
        ax.set_yticks(np.arange(len(functions_list)))
        ax.set_xticks(np.arange(len(optimizers_list)))
        # Set labels for x and y axis
        ax.set_yticklabels(functions_list)
        ax.set_xticklabels(optimizers_list)

        plt.setp(ax.get_xticklabels(),
                 rotation=45,
                 ha="right",
                 rotation_mode="anchor")
        ax.set_title(opt + ' Results')
        fig.tight_layout()
        if (not os.path.exists('results/plots_' + str(iterations))):
            os.mkdir('results/plots_' + str(iterations))
        plt.savefig("results/plots_" + str(iterations) + '/' + opt + ".png",
                    bbox_inches='tight')


# Function which returns 2 lists : opt_names which contains list of optimizers and
# opt_config which contains different configurations for these optimizers
def convert_to_list(opt_combinations):
    opt_names, opt_config = [], []
    for key in opt_combinations:
        for par in opt_combinations[key]:
            opt_names.append(key)
            opt_config.append(par)
    return opt_names, opt_config


def main():
    global func_params, opt_params, func_constraints
    # Checking if result file exists earlier or not
    if not os.path.exists("results"):
        os.mkdir("results")
    results = {}
    print('generating functions....')
    functions_list = [x for x in dir(functions) if x.find('loss_') >= 0]
    print('generating optimizers....')
    opt_combinations, total_optimizers = generate_optimizers(opt_params)
    opt_names, opt_config = convert_to_list(opt_combinations)

    num_runs = 10
    print('generating function constraints...')
    func_parameter_combinations = generate_functions(functions_list,
                                                     func_params,
                                                     func_constraints)
    add_result_to_json(results, opt_names, opt_config, functions_list,
                       func_parameter_combinations)

    # generate initial states
    initial_tensors = generate_initial(opt_names, func_parameter_combinations,
                                       num_runs)

    n_workers = 8 if multiprocessing.cpu_count() > 4 else 4
    print(n_workers)
    with multiprocessing.Pool(processes=n_workers) as pool:
        for i in range(num_runs):
            print('Running iteration %d ....' % i)
            #running every combination of optimizers on every function
            t1 = time.time()
            for j in range(len(functions_list)):
                print(functions_list[j])
                soln = pool.starmap(
                    run_function,
                    zip(repeat(functions_list[j]), opt_names,
                        repeat(func_parameter_combinations), opt_config,
                        initial_tensors[i][j]))
                add_soln_to_results(
                    results, functions_list[j],
                    func_parameter_combinations[functions_list[j]], soln,
                    opt_names, opt_config)
            print('time taken = ', time.time() - t1)
    # Writing results to a json object
    with open('results/opt_results_' + str(iterations) + '.json',
              'w+') as file:
        json.dump(results, file, indent=4)
    print('plotting results...')
    plot_results()


if __name__ == '__main__':
    main()
