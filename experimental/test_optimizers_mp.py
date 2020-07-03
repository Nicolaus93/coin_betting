import numpy as np
import torch
import optimal_pytorch
import json
import os
import time
import multiprocessing
import argparse
from itertools import repeat
from plot_results import plot_results
from Tests import functions
from Data import config
# For reproducibility.
torch.manual_seed(1)

# file paths containing different parameters for optimizers (lr, momentum etc) and function constraints(xs and xe).
opt_params_path = '../Data/opt_params.json'
func_constraints_path = '../Data/func_constraints.json'
results_path = '../Results/unit_tests'

# Loading configuration variables(num_runs, n_iterations etc).
conf = config.Unit_Test_Config()


def generate_optimizers(params):
    """
    Generates a list of all possible hyperparameter combinations for optimizer as defined in params.
    """
    opt_combinations = {}
    counter = 0
    for key in params:
        # Create an empty array if that key is empty [will happen in case of first key(here Adam)].
        if key not in opt_combinations:
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


def generate_functions(functions_list, func_constraints, conf):
    """
    Generates random parameter combination for functions to be used in the experiments.
    functions_list = A list of all the functions
    func_params = a list of all the parameters every function requires(ex: xs, xe, mu)
    func_constraints = Functions which have constraints for the input space(have a specific minimum in that domain).
    conf = configuration file containing different config vars(num_iterations etc).
    """
    scaling = 10
    dic = {}
    for name in functions_list:
        dic[name] = {}
        if ('params' in func_constraints[name]):
            for i in range(conf.num_func_variations):
                for param in func_constraints[name]['params']:
                    if (not param in dic[name]):
                        dic[name][param] = []
                    temp = torch.rand(1, dtype=torch.float) * (
                        func_constraints[name]['scaling'][1] -
                        func_constraints[name]['scaling'][0]
                    ) + func_constraints[name]['scaling'][0]
                    dic[name][param].append(temp)
                if ('xs' in dic[name]):
                    if (dic[name]['xs'][i] > dic[name]['xe'][i]):
                        dic[name]['xs'][i], dic[name]['xe'][i] = dic[name][
                            'xe'][i], dic[name]['xs'][i]
    return dic


"""
Initializes results object with every entry, The results is finally in the format :
 { Opt_name1: { Opt_parameters1:{ function1:{ func_config1{
                                                    x_initial : [...],
                                                    x_soln : [...],
                                                    x_optimal: [...],
                                                    color: [...],
                                                    iterations: [...]
                                                },
                                                func_config2{...}
                                            },
                                function2: {...}
                               },
                Opt_parameters2:{...}
               },
    Opt_name2:{...}
 }
"""


def add_result_to_json(results, opt_name, opt_params, functions_list,
                       func_params, conf):
    """Function to add populate empty results onject to the format described above.
    results : empty json object.
    opt_name : array containing name of optimizers.
    opt_params : var containing different config for opt_params.
    functions_list : var containing names of all functions.
    func_params : var containing different set of parameters for all functions.
    conf : variable containing config parameters.
    """
    for i in range(len(opt_params)):
        # Check if optimizer is already present in dict object
        if (not opt_name[i] in results):
            results[opt_name[i]] = {}
        # create an entry for a specific set of hyperparameters for the optimizer
        str1 = ''
        for key in opt_params[i]:
            str1 += (key + '|' + str(opt_params[i][key]) + '|')
        results[opt_name[i]][str1] = {}
        # Create an entry for the specific parameters for the function
        for func_name in functions_list:
            results[opt_name[i]][str1][func_name] = {}
            # creating entries for our starting point, optimal solution
            # minima, color and iterations to reach that point
            temp_json = {}
            temp_json['x_optimal'] = []
            temp_json['x_soln'] = []
            temp_json['x_initial'] = []
            temp_json['color'] = []
            temp_json['iterations'] = []
            if (not bool(func_params[func_name])):
                results[opt_name[i]][str1][func_name] = temp_json
            else:
                for j in range(conf.num_func_variations):
                    str2 = ''
                    for param in func_params[func_name]:
                        temp = func_params[func_name][param][j].data.numpy(
                        ).reshape(1)
                        str2 += (param + '|' + str(temp[0]) + '|')
                    results[opt_name[i]][str1][func_name][str2] = temp_json


def get_limits(func_constraints, func_combinations, key, j):
    """Helper function which for any function returns the domain of the function [xstart, xend].
    func_constraints : variable containing constraints for generating function config("limit" key is used to generate the domain).
    func_combinations : variable containing function configurations for all functions.
    key : name of the function
    j : jth iteration number.
    """
    start = func_constraints[key]['limit'][0]
    end = func_constraints[key]['limit'][1]
    if (isinstance(start[0], str)):
        if (isinstance(start[1], str)):
            xs = func_combinations[key][
                start[0]][j] + start[2] * func_combinations[key][start[1]][j]
        else:
            xs = func_combinations[key][start[0]][j] + start[2] * start[1]
    else:
        xs = start[0] + start[2] * start[1]

    if (isinstance(end[0], str)):
        if (isinstance(end[1], str)):
            xe = func_combinations[key][
                end[0]][j] - end[2] * func_combinations[key][end[1]][j]
        else:
            xe = func_combinations[key][end[0]][j] - end[2] * end[1]
    else:
        xe = end[0] - end[2] * end[1]
    return xs, xe


def generate_initial(optimizer_name, functions_combinations, conf,
                     func_constraints):
    """
    This function generates the initial value for every function-optimizer combination
    for some num_runs between [xs, xe] for every function.
    """
    init_tensors = []
    for i in range(conf.num_runs):
        init_run = []
        for key in functions_combinations:
            func_temp = []
            if (bool(functions_combinations[key])):
                num_variations = conf.num_func_variations
            else:
                num_variations = 1
            for j in range(num_variations):
                xs, xe = get_limits(func_constraints, functions_combinations,
                                    key, j)
                combo = torch.rand(1, dtype=torch.float) * (xe - xs) + xs
                func_temp.append(combo)
            func_run = []
            for _ in range(len(optimizer_name)):
                temp = [ele.clone() for ele in func_temp]
                for ele in temp:
                    ele.requires_grad_()
                func_run.append(temp)
            init_run.append(func_run)
        init_tensors.append(init_run)
    return init_tensors


def compare_results(optimal, obtained, calc_diff=False):
    """
    Function to assign colors based on the difference between values.
    """
    a = float(optimal)
    b = float(obtained)
    c = abs(a - b)
    if (calc_diff):
        return (c)
    else:
        if (c <= 0.01):
            return 'g'
        elif (c >= 0.01 and c <= 0.05):
            return 'b'
        else:
            return 'r'


def run_function(func_name, opt_name, func_params, opt_params, initial,
                 func_constraints):
    """
    runs an optimizer instance on a function for n iterations
    func_name = which function is used
    opt_name = which optimizer is used
    func_params = the parameters of that specific function
    opt_params = hyper parameters for that specific optimizer
    initial = the initial value to optimize.
    """
    global conf
    init_arr, minima_arr, final_arr, iteration_arr = [], [], [], []
    if ('params' in func_constraints[func_name]):
        num_variations = conf.num_func_variations
    else:
        num_variations = 1
    for i in range(num_variations):
        temp = {}
        if ('params' in func_constraints[func_name]):
            for param in func_constraints[func_name]['params']:
                temp[param] = func_params[func_name][param][i]
        loss_function = getattr(functions, func_name)(temp)
        minima = loss_function.get_minima()
        x = initial[i]
        init_arr.append(str(x.clone().data.numpy()[0]))
        # Runs loss_function on optimizer for some iterations
        optimizer = getattr(optimal_pytorch, opt_name)([x], **opt_params)
        counter = 0
        for _ in range(int(func_constraints[func_name]['max_iterations'])):
            optimizer.zero_grad()
            loss = loss_function.forward(x)
            loss.backward()
            optimizer.step()
            counter += 1
            if (compare_results(x, minima) == 'g'
                    or compare_results(x, minima) == 'b'):
                break
        minima_arr.append(str(minima.data.numpy().reshape(1)[0]))
        final_arr.append(str(x.data.numpy().reshape(1)[0]))
        iteration_arr.append(counter)
    return init_arr, minima_arr, final_arr, iteration_arr


def add_soln_to_results(results, func_name, func_params, soln, opt_names,
                        opt_config, func_constraints):
    """
    Function which adds solutions generated during the optimization process to a json object.
    results = The json object which will contains all the results
    func_name = the name of the function
    func_params = the parameters of that function,
        soln contains the final values obtained in the optimization
        process for an optimizer on a function
    opt_names = list of all optimizers
    opt_config = config for every optimizer in opt_names
    """
    global conf
    for i in range(len(opt_names)):
        opt_name = opt_names[i]
        opt_conf = opt_config[i]
        str1 = ''
        for key in opt_conf:
            str1 += (key + '|' + str(opt_conf[key]) + '|')
        if ('params' in func_constraints[func_name]):
            num_variations = conf.num_func_variations
        else:
            num_variations = 1
        for j in range(num_variations):
            str2 = ''
            for key in func_params[func_name]:
                str2 += (key + '|' + str(
                    func_params[func_name][key][j].data.numpy().reshape(1)[0])
                         + '|')
            if (len(str2) == 0):
                results[opt_name][str1][func_name]['x_initial'].append(
                    str(soln[i][0][j]))
                results[opt_name][str1][func_name]['x_optimal'].append(
                    str(soln[i][1][j]))
                results[opt_name][str1][func_name]['x_soln'].append(
                    str(soln[i][2][j]))
                results[opt_name][str1][func_name]['color'].append(
                    str(compare_results(soln[i][1][j], soln[i][2][j])))
                results[opt_name][str1][func_name]['iterations'].append(
                    str(soln[i][3][j]))
            else:
                results[opt_name][str1][func_name][str2]['x_initial'].append(
                    str(soln[i][0][j]))
                results[opt_name][str1][func_name][str2]['x_optimal'].append(
                    str(soln[i][1][j]))
                results[opt_name][str1][func_name][str2]['x_soln'].append(
                    str(soln[i][2][j]))
                results[opt_name][str1][func_name][str2]['color'].append(
                    str(compare_results(soln[i][1][j], soln[i][2][j])))
                results[opt_name][str1][func_name][str2]['iterations'].append(
                    str(soln[i][3][j]))


def give_next_opt(opt_combinations, counter):
    """
    iterator like function which gives the next optimizer and its hyperparameters.
    """
    temp = counter
    for opt_key in opt_combinations:
        if (temp >= len(opt_combinations[opt_key])):
            temp = temp - len(opt_combinations[opt_key])
            continue
        else:
            return opt_key, opt_combinations[opt_key][temp]


def convert_to_list(opt_combinations):
    """
    Function which returns 2 lists : opt_names which contains list of optimizers and
    opt_config which contains different configurations for these optimizers
    """
    opt_names, opt_config = [], []
    for key in opt_combinations:
        for par in opt_combinations[key]:
            opt_names.append(key)
            opt_config.append(par)
    return opt_names, opt_config


def run():
    global conf, opt_params_path, func_constraints_path, results_path
    # making a directory to store results and loading env variables
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
    with open(opt_params_path) as file:
        opt_params = json.load(file)
    with open(func_constraints_path) as file:
        func_constraints = json.load(file)

    results = {}
    print('generating functions....')
    functions_list = [x for x in dir(functions) if x.find('loss_') >= 0]
    print('generating optimizers....')
    opt_combinations, total_optimizers = generate_optimizers(opt_params)
    opt_names, opt_config = convert_to_list(opt_combinations)

    print('generating function constraints...')
    func_parameter_combinations = generate_functions(functions_list,
                                                     func_constraints, conf)
    add_result_to_json(results, opt_names, opt_config, functions_list,
                       func_parameter_combinations, conf)
    # generate initial states
    initial_tensors = generate_initial(opt_names, func_parameter_combinations,
                                       conf, func_constraints)
    n_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=n_workers) as pool:
        for i in range(conf.num_runs):
            print('Running iteration %d ....' % i)
            # running every combination of optimizers on every function
            t1 = time.time()
            for j in range(len(functions_list)):
                t2 = time.time()
                print(functions_list[j])
                soln = pool.starmap(
                    run_function,
                    zip(repeat(functions_list[j]), opt_names,
                        repeat(func_parameter_combinations), opt_config,
                        initial_tensors[i][j], repeat(func_constraints)))
                add_soln_to_results(results, functions_list[j],
                                    func_parameter_combinations, soln,
                                    opt_names, opt_config, func_constraints)
                print('Function running time = ', time.time() - t2)
            print('time taken = ', time.time() - t1)
    # Writing results to a json object
    with open(results_path + '/logs' + '.json', 'w+') as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot results of unit tests.')
    parser.add_argument('--plot',
                        dest='plot',
                        action='store_true',
                        default=False,
                        help="whether to plot results (default=False).")

    args = parser.parse_args()
    plot = args.plot
    run()
    if plot:
        print('plotting results...')
        plot_results(results_path, iterations, True)
