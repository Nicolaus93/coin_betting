import numpy as np
import torch
import optimal_pytorch
import functions
import json
import os
import math
import time
from collections import Counter
from matplotlib import pyplot as plt
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

# Color codes.
color_rgb = {
    'r': [194, 24, 7],
    'g': [41, 171, 135],
    'b': [15, 82, 186],
    'y': [255, 255, 0],
    'o': [255, 170, 29],
    'p': [254, 127, 156]
}
# Number of iterations to run one function-optimizer combination on.
iterations = 1000


# Generates a list of all possible combinations for optimizer.
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
                # Create n copies of the original list.
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


# Generates a random parameter combination for functions to be used in the experiments.
def generate_functions(functions_list, func_params, func_constraints):
    scaling = 10
    dic = {}
    vars = torch.rand(len(func_params), dtype=torch.float) * scaling
    for name in functions_list:
        dic[name] = {}
        for i in range(len(func_params)):
            if (name in func_constraints):
                xs = func_constraints[name]['xs']
                xe = func_constraints[name]['xe']
                if (not (func_params[i] == 'xs' or func_params[i] == 'xe')):
                    if (vars[i] > xe or vars[i] < xs):
                        dic[name][func_params[i]] = torch.rand(
                            1, dtype=torch.float) * (xe - xs) + xs
                    else:
                        dic[name][func_params[i]] = vars[i]
                elif (func_params[i] == 'xs'):
                    dic[name][func_params[i]] = torch.tensor(xs)
                elif (func_params[i] == 'xe'):
                    dic[name][func_params[i]] = torch.tensor(xe)
            else:
                dic[name][func_params[i]] = vars[i]
        #storing xs and xe so that we generate our initial x between these values.
        if (not name in func_constraints):
            func_constraints[name] = {}
            func_constraints[name]['xs'] = np.array(
                vars[2].data.numpy()).reshape(1)[0]
            func_constraints[name]['xe'] = np.array(
                vars[3].data.numpy()).reshape(1)[0]
    return dic


# generates a list of all the initial tensors for every function-optimizer combination to be used in every iteration.
def generate_initial(total_optimizers, functions_combinations, num_runs):
    init_tensors = []
    for i in range(num_runs):
        init_run = []
        for key in functions_combinations:
            xs = functions_combinations[key]['xs']
            xe = functions_combinations[key]['xe']
            combo = [
                torch.rand(1, dtype=torch.float) * (xe - xs) + xs
                for _ in range(total_optimizers)
            ]
            for ele in combo:
                ele.requires_grad_()
            init_run.append(combo)
        init_tensors.append(init_run)
    return init_tensors


# Runs an optimizer instance on a function. Func_params and opt_params are dict objects containing parameters.
def run_function(func_name, opt_name, func_params, opt_params,
                 func_constraints, initial):
    global iterations
    loss_function = getattr(functions, func_name)(func_params[func_name])
    # Generates a random value between xs and xe for a function.
    xs = func_constraints[func_name]['xs']
    xe = func_constraints[func_name]['xe']
    x = initial
    func_params[func_name]['initial'] = str(x.clone().data.numpy()[0])
    # Runs loss_function on optimizer for some iterations.
    optimizer = getattr(optimal_pytorch, opt_name)([x], **opt_params)
    for i in range(iterations):
        optimizer.zero_grad()
        loss = loss_function.forward(x)
        loss.backward()
        optimizer.step()
    # Storing results in a json object.
    minima = loss_function.get_minima()
    if (torch.is_tensor(minima)):
        func_params[func_name]['min'] = str(minima.data.numpy().reshape(1)[0])
    else:
        func_params[func_name]['min'] = str(minima)
    func_params[func_name]['soln'] = str(x.data.numpy().reshape(1)[0])


"""Color coding used to plot the graphs :
    red = if diff between reached and optimal solution is more than 0.5.
    blue = if diff between reached and optimal solution is between 0.1 and 0.5.
    green = if diff between reached and optimal solution is less than 0.1.

    red + blue => pink, when red and blue are obtained in equal proportions in n runs.
    red + green => yellow, when red and green are obtained in equal proportions in n runs.
    green + blue => orange, when green and blue are obtained in equal proportions in n runs.

    in all other cases the color which occurs max number of times is plotted.
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


# For cases where the max colors are same.
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


# Iterator like function which gives the next optimizer and its properties.
def give_next_opt(opt_combinations, counter):
    temp = counter
    for opt_key in opt_combinations:
        if (temp >= len(opt_combinations[opt_key])):
            temp = temp - len(opt_combinations[opt_key])
            continue
        else:
            return opt_key, opt_combinations[opt_key][temp]


# Function which adds the results generated in an experiment run to the results json object.
def add_result_to_json(results, i, opt_name, opt_params, func_name,
                       func_params):
    # Check if optimizer is already present in dict object.
    if (not opt_name in results):
        results[opt_name] = {}
    # create an entry for a specific set of hyperparameters for the optimizer.
    str1 = ''
    for param in opt_params:
        str1 += (param + '|' + str(opt_params[param]) + '|')
    if (not str1 in results[opt_name]):
        results[opt_name][str1] = {}
    # Create an entry for the specific parameters for the function (will be same, since created using the same seed).
    str2 = ''
    str2 += (func_name + '|')
    for param in func_params:
        if (not (param == 'min' or param == 'soln' or param == 'initial')):
            temp = func_params[param].data.numpy().reshape(1)
            str2 += (param + '|' + str(temp[0]) + '|')
    #creating entries for our starting point, optimal solution, solution after n steps and color assigned.
    if (not str2 in results[opt_name][str1]):
        results[opt_name][str1][str2] = {}
        results[opt_name][str1][str2]['x_optimal'] = []
        results[opt_name][str1][str2]['x_soln'] = []
        results[opt_name][str1][str2]['x_initial'] = []
        results[opt_name][str1][str2]['color'] = []

    results[opt_name][str1][str2]['x_optimal'].append(func_params['min'])
    results[opt_name][str1][str2]['x_soln'].append(func_params['soln'])
    results[opt_name][str1][str2]['x_initial'].append(func_params['initial'])
    results[opt_name][str1][str2]['color'].append(
        compare_results(func_params['min'], func_params['soln']))


#returns an array containing count of each colors.
def most_frequent(List):
    return Counter(List)


# Function to Plots results which were saved in results/opt_results_[n-iterations].json file.
def plot_results():
    global color_rgb
    with open('results/opt_results_' + str(iterations) + '.json') as file:
        results = json.load(file)
    compiled = {}
    for opt in results:
        # Selecting the most common occuring color of the n rounds.
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
            #For every function ex - loss_gaussian, loss_absolute.
            for functions in compiled[opt][config]:
                pos = functions.find('|')
                color_val = compiled[opt][config][functions]
                # Create only one copy of function name in functions_list.
                if (not functions[:pos] in functions_list):
                    functions_list.append(functions[:pos])
                col_temp.append(color_val)
            color_arr.append(col_temp)

        # reshaping it to n X m, where n = number of functions, m = number of optimizers.
        color_arr = np.transpose(np.array(color_arr)).reshape(
            len(functions_list), len(optimizers_list))
        rgb_arr = np.array([[color_rgb[ele] for ele in row]
                            for row in color_arr])
        # plotting on heatmap and saving.
        fig, ax = plt.subplots(figsize=(15, 40))
        plt.imshow(rgb_arr)
        # Ticks are just number of boxes.
        ax.set_yticks(np.arange(len(functions_list)))
        ax.set_xticks(np.arange(len(optimizers_list)))
        # Set labels for x and y axis.
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
        plt.savefig("results/plots_" + str(iterations) + "/" + opt + ".png",
                    bbox_inches='tight')


def main():
    global func_params, opt_params
    # Checking if result file exists earlier or not.
    if not os.path.exists("results"):
        os.mkdir("results")
    results = {}
    print('generating functions....')
    functions_list = [x for x in dir(functions) if x.find('loss_') >= 0]
    print('generating optimizers....')
    opt_combinations, total_optimizers = generate_optimizers(opt_params)
    num_runs = 10
    print('generating function constraints...')
    func_parameter_combinations = generate_functions(functions_list,
                                                     func_params,
                                                     func_constraints)
    print('generating Initial values for experiments...')
    initial_tensors = generate_initial(total_optimizers,
                                       func_parameter_combinations, num_runs)
    for i in range(num_runs):
        t1 = time.time()
        print('Running iteration %d ....' % i)
        # Running every combination of optimizers on every function.
        for j in range(total_optimizers):
            opt_name, opt_params = give_next_opt(opt_combinations, j)
            for k in range(len(functions_list)):
                run_function(functions_list[k], opt_name,
                             func_parameter_combinations, opt_params,
                             func_constraints, initial_tensors[i][k][j])
                add_result_to_json(
                    results, i, opt_name, opt_params, functions_list[k],
                    func_parameter_combinations[functions_list[k]])
        print('time taken = ', time.time() - t1)
    # Writing results to a json object.
    with open('results/opt_results_' + str(iterations) + '.json',
              'w+') as file:
        json.dump(results, file, indent=4)
    print('plotting results...')
    plot_results()


if __name__ == '__main__':
    main()
