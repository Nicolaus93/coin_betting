"""Color coding used to plot the graphs
red = if diff between reached and optimal solution is more than 0.5
blue = if diff between reached and optimal solution is between 0.1 and 0.5
green = if diff between reached and optimal solution is less than 0.1

red + blue => pink, when red and blue are obtained in equal proportions in n runs
red + green => yellow, when red and green are obtained in equal proportions in n runs
green + blue => orange, when green and blue are obtained in equal proportions in n runs

in all other cases the color which occurs max number of times is plotted
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from typing import Sequence, Mapping
from collections import Counter


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


def plot_results(iterations: int, save: bool = False) -> None:
    """
    Function to Plots results which were saved in results/opt_results.json file
    """
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
            if (not os.path.exists('results/plots_' + str(iterations))):
                os.mkdir('results/plots_' + str(iterations))
            plt.savefig("results/plots_" + str(iterations) + '/' + opt + ".png",
                        bbox_inches='tight')
        else:
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot results of unit tests.')
    parser.add_argument('-it', dest='iterations', type=int, default=500,
                        help='select the experiment to plot.')
    parser.add_argument('--save', dest='save', action='store_true',
                        default=False, help="whether to save plots (default=False).")

    args = parser.parse_args()
    iterations = args.iterations
    save = args.save
    plot_results(iterations, save=save)
