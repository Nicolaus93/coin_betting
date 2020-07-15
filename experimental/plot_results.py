import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from typing import Sequence, Mapping
from collections import Counter
import optimal_pytorch.functions as functions
import glob
from Data.scripts import config
import torch
from torch import Tensor

"""Color coding used to plot the graphs
red = if diff between reached and optimal solution is more than 0.5
blue = if diff between reached and optimal solution is between 0.1 and 0.5
green = if diff between reached and optimal solution is less than 0.1

red + blue => pink, when red and blue are obtained in equal proportions in n runs
red + green => yellow, when red and green are obtained in equal proportions in n runs
green + blue => orange, when green and blue are obtained in equal proportions in n runs

in all other cases the color which occurs max number of times is plotted
"""
conf = config.Unit_Test_Config()

# Color codes for plotting graphs.
COLOR_RGB = {
    'r': [194, 24, 7],
    'g': [41, 171, 135],
    'b': [15, 82, 186],
    'y': [255, 255, 0],
    'o': [255, 170, 29],
    'p': [254, 127, 156]
}


def most_frequent(List: Sequence[int]) -> Mapping[str, int]:
    """
    Helper function returning an array containing count of each entry.
    """
    return Counter(List)

def compare_results(optimal: str, obtained: str, func=None)-> Tensor:
    """
    Function to assign colors based on the difference between values.
    """
    a = torch.tensor(float(optimal))
    b = torch.tensor(float(obtained))
    if (func is None):
        c = torch.abs(a - b)
    else:
        c = torch.abs(func.forward(a) - func.forward(b))

    if (c <= 0.01):
        return 'g'
    elif (c >= 0.01 and c <= 0.05):
        return 'b'
    else:
        return 'r'

def mix_color(counter: Sequence[tuple]) -> str:
    """
    for cases where the max colors are same.
    """
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

def calc_metric(loss_function , x_opti: Tensor, x_sol: Tensor, metric: str)-> Tensor:
    """Helper function which calculates difference metric.
    loss_function : the loss function we use.
    x_opti : the optimal value for that function (minima).
    x_sol : the value we obtained after some rounds of optimization.
    metric : L1 or suboptimal
    """
    if(metric=='L1' or metric=='l1'):
        return torch.abs(x_opti - x_sol)
    elif(metric.lower()=='suboptimal'):
        return torch.abs(loss_function.forward(x_opti) - loss_function.forward(x_sol))


def generate_colors(results: dict, func_name: str, conf: dict, metric: str, func_variation: str = None) -> None:
    """Helper function which assigns color for the obtained vs optimal value based on a difference metric.
    results : the result object where all the results will be stored.
    func_name : name of the function.
    conf : config object which contains different program parameters.
    metric : the difference metric (L1 or suboptimal).
    func_variation : for functions which have coefficients.
    """
    # For functions which have different coefficients.
    if(func_variation is not None):
        results[func_name][func_variation]['color'] = []
        # Obj contains the coefficients which are passed to the function to instantiate it.
        obj = {}
        func_variation_arr = func_variation.split('|')[:-1]
        for i in range(len(func_variation_arr)//2):
            obj[func_variation_arr[2*i]] = float(func_variation_arr[2*i + 1])
        func = getattr(functions.loss_functions_1d, func_name)(obj)
        # Now we just run function for different runs of our experiment
        for i in range(conf.num_runs):
            x_sol = results[func_name][func_variation]['x_soln'][i]
            x_opti = results[func_name][func_variation]['x_optimal'][i]
            if(metric=='L1' or metric=='l1'):
                results[func_name][func_variation]['color'].append(compare_results(x_sol, x_opti))
            else:
                results[func_name][func_variation]['color'].append(compare_results(x_sol, x_opti, func))
    else:
        results[func_name]['color'] = []
        obj = {}
        func = getattr(functions.loss_functions_1d, func_name)(obj)
        # Now we just run function for different runs of our experiment
        for i in range(conf.num_runs):
            x_sol = results[func_name]['x_soln'][i]
            x_opti = results[func_name]['x_optimal'][i]
            if(metric=='L1' or metric=='l1'):
                results[func_name]['color'].append(compare_results(x_sol, x_opti))
            else:
                results[func_name]['color'].append(compare_results(x_sol, x_opti, func))


def plot_results(metric: str, save: bool = False) -> None:
    """
    Function to Plots results which were saved in results/opt_results.json file
    """
    global conf
    with open('../Data/func_constraints.json') as file:
        func_constraints = json.load(file)
    
    results_path = '../Results/unit_tests'
    all_log_files = glob.glob(results_path + '/logs_*.json')
    results = {}
    for log_file in all_log_files:
        with open(log_file) as file:
            temp = json.load(file)
            pos = log_file.find('logs_')
            key = log_file[pos+5:-5]
            results[key] = temp

    compiled = {}
    for opt in results:
        # Selecting the most common occuring color of the n rounds
        compiled[opt] = {}
        for config in results[opt]:
            if(config != 'curr_run'):
                compiled[opt][config] = {}
                for func_name in results[opt][config]:
                    # Since log files will have different function variants (coeficients), we calculate top color for every variant.
                    if('params' in func_constraints[func_name]):
                        compiled[opt][config][func_name] = {}
                        for func_variation in results[opt][config][func_name]:
                            # Calculate colors based on the metric provided.
                            generate_colors(results[opt][config], func_name, conf, metric,func_variation)
                            # Selects top 2 frequent colors
                            counter = most_frequent(
                                results[opt][config][func_name][func_variation]['color']).most_common(2)
                            # For cases where 2 colors have the same frequency
                            if (len(counter) > 1):
                                compiled[opt][config][func_name][func_variation] = mix_color(counter)
                            else:
                                compiled[opt][config][func_name][func_variation] = counter[0][0]
                        # This choses the most common occuring color among all the different function variant.
                        temp = []
                        for func_variation in results[opt][config][func_name]:
                            temp.append(compiled[opt][config][func_name][func_variation])
                        # Choses the majority color.
                        counter2 = most_frequent(temp).most_common(1)
                        compiled[opt][config][func_name] = counter2[0][0]
                    else:
                        # For functions which don't have any variants.
                        generate_colors(results[opt][config], func_name, conf, metric)
                        counter = most_frequent(
                            results[opt][config][func_name]['color']).most_common(2)
                        if (len(counter) > 1):
                            compiled[opt][config][func_name] = mix_color(counter)
                        else:
                            compiled[opt][config][func_name] = counter[0][0]

    """Creates an n X m array where n = number of functions and m = optimizer
    combinations and array values contain the data in the object 'compiled',
    Finally color_arr contains color values like : 'r', 'g', 'b' etc.
    and rgb_arr contains a conversion of these colors into their rgb values as per color_rgb object.
    """
    # For every optimizer ex - Adam, SGD etc.
    for opt in compiled:
        optimizers_list = []
        functions_list = []
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
            # For every function ex - loss_gaussian, loss_absolute
            for function_name in compiled[opt][config]:
                pos = function_name.find('|')
                color_val = compiled[opt][config][function_name]
                # Create only one copy of function name in functions_list
                if (not function_name[:pos] in functions_list):
                    functions_list.append(function_name[:pos])
                col_temp.append(color_val)
            color_arr.append(col_temp)
        
        # reshaping it to n X m, where n = number of functions, m = number of optimizers
        color_arr = np.transpose(np.array(color_arr)).reshape(
            len(functions_list), len(optimizers_list))
        rgb_arr = np.array([[COLOR_RGB[ele] for ele in row]
                            for row in color_arr])
        
        # plotting on heatmap and saving
        # fig, ax = plt.subplots(figsize=(15, 40))
        fig, ax = plt.subplots()
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
        if save:
            if (not os.path.exists(results_path + '/plots_' +
                                   str(iterations))):
                os.makedirs(results_path + '/plots_' + str(iterations),
                            exist_ok=True)
            plt.savefig(results_path + '/plots_' + str(iterations) + '/' +
                        opt + ".png",
                        bbox_inches='tight')
        else:
            plt.show()


def interpret_results(metric: str)-> None:
    global conf
    with open('../Data/func_constraints.json') as file:
        func_constraints = json.load(file)
    
    with open('../Data/func_constraints.json') as file:
        func_constraints = json.load(file)

    results_path = '../Results/unit_tests'
    all_log_files = glob.glob(results_path + '/logs_*.json')

    results = {}
    for log_file in all_log_files:
        with open(log_file) as file:
            temp = json.load(file)
            pos = log_file.find('logs_')
            key = log_file[pos+5:-5]
            results[key] = temp
    
    # compiled stores the suboptimality gap between x_initial and x_soln for all optimizer configurations.
    compiled = {}
    for opt in results:
        compiled[opt] = {}
        for config in results[opt]:
            # Every log file has a key curr_run which won't be used here.
            if(config != 'curr_run'):
                compiled[opt][config] = {}
                # For every function
                for func_name in results[opt][config]:
                    compiled[opt][config][func_name] = []
                    # Functions which have params have different combinations of these params.
                    if('params' in func_constraints[func_name]):
                        for func_variation in results[opt][config][func_name]:
                            temp = func_variation.split('|')[:-1]
                            # obj contains all the coefficients used to instantiate a function.
                            obj = {}
                            for i in range(len(temp)//2):
                                obj[temp[2*i]] = float(temp[2*i + 1])

                            loss_function = getattr(functions.loss_functions_1d, func_name)(obj)
                            for i in range(conf.num_runs):
                                x_opti = torch.tensor(float(results[opt][config][func_name][func_variation]['x_optimal'][i]))
                                x_sol = torch.tensor(float(results[opt][config][func_name][func_variation]['x_soln'][i]))
                                get_metric_val = calc_metric(loss_function, x_opti, x_sol, metric)
                                compiled[opt][config][func_name].append(get_metric_val.data.numpy().reshape(1)[0])
                    else: 
                    # for functions which don't have any coefficients
                        obj = {}
                        loss_function = getattr(functions.loss_functions_1d, func_name)(obj)
                        for i in range(conf.num_runs):
                            x_opti = torch.tensor(float(results[opt][config][func_name]['x_optimal'][i]))
                            x_sol = torch.tensor(float(results[opt][config][func_name]['x_soln'][i]))
                            get_metric_val = calc_metric(loss_function, x_opti, x_sol, metric)
                            compiled[opt][config][func_name].append(get_metric_val.data.numpy().reshape(1)[0])
    
    # Final will contain all the results for comparisons between SGD and different optimizers.
    final = {}
    for opt in compiled:
        final[opt] = {}
        for ele in dir(functions):
            if(ele.find('loss_')>=0):
                final[opt][ele] = {}

    # Finding the lowest metric(suboptimality gap, L1 norm) for SGD (baseline).
    for opt_config in compiled['SGD']:
        for func_name in compiled['SGD'][opt_config]:
            temp_arr = compiled['SGD'][opt_config][func_name]
            temp_arr.sort()
            if('best' not in final['SGD'][func_name]):
                final['SGD'][func_name]['best'] = temp_arr[0]
                final['SGD'][func_name]['config'] = opt_config
            else:
                if(temp_arr[0]<final['SGD'][func_name]['best']):
                    final['SGD'][func_name]['best'] = temp_arr[0]
                    final['SGD'][func_name]['config'] = opt_config
    
    for opt in compiled:
        # We have already calculated for SGD.
        if(opt != 'SGD'):
            for config in compiled[opt]:
                for func_name in compiled[opt][config]:
                    # compiled for every opt config key contains just a big array of all the metric values(L1 or suboptimal)
                    # that we have calculated for every function.
                    for gap in compiled[opt][config][func_name]:
                        if('passed' not in final[opt][func_name]):
                            final[opt][func_name]['passed'] = 0
                            final[opt][func_name]['total'] = 0
                        # If performance is better than the best SGD, it has passed the test.
                        if(gap<=final['SGD'][func_name]['best']):
                            final[opt][func_name]['passed'] += 1
                        final[opt][func_name]['total'] += 1
    for opt in final:
        if(opt != 'SGD'):
            for func_name in final[opt]:
                print(opt+' has passed '+ str(final[opt][func_name]['passed']) + '/'+ str(final[opt][func_name]['total']) + ' tests for '+ func_name + ' function.')
                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot results of unit tests.')
    parser.add_argument('--plot',
                    dest='plot',
                    action='store_true',
                    default=False,
                    help="whether to plot results")

    parser.add_argument('--interpret',
                        dest='interpret',
                        action='store_true',
                        default=False,
                        help="whether to interpret results (in text form)?")
    
    parser.add_argument('--metric',
                        dest='metric',
                        required=True,
                        help="Use which metric to generate results (L1 or suboptimal)?")

    parser.add_argument('--save',
                        dest='save',
                        action='store_true',
                        default=False,
                        help="whether to save plots (default=False).")

    args = parser.parse_args()
    if(args.plot):
        plot_results(args.metric, args.save)
    elif(args.interpret):
            interpret_results(args.metric)
