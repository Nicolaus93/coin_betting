import torch
import optimal_pytorch.optim as optimal_pytorch
import json

# Gives a list of all optimizers in library.
all_optimizers = optimal_pytorch.__all__
optim_params = {}
for ele in all_optimizers:
        optim = getattr(optimal_pytorch, ele)
        #check if that optimizer has the member function grid_search_params.
        if('grid_search_params' in dir(optim)):
            # A dummy tensor for instantiating the optim object.
            x = torch.tensor(1.0, requires_grad=True)
            grid_params = optim([x]).grid_search_params()
            optim_params[ele] = {}
            for key in grid_params.keys():
                if('use' in grid_params[key]):
                    # If we have the 'use' keyword, only use the values before that.
                    optim_params[ele][key] = grid_params[key][:-1]
                elif('gen' in grid_params[key]):
                    # If we have gen keyword, we find the values based on lower and upper limit. 
                    # (currently, they are incremented in powers of 10).
                    optim_params[ele][key] = []
                    lower = grid_params[key][0]
                    upper = grid_params[key][1]
                    nb_values = grid_params[key][len(grid_params[key])-1]
                    curr = lower
                    for i in range(nb_values):
                        optim_params[ele][key].append(curr)
                        curr =  curr * 10
                elif(type(grid_params[key][0])==list):
                    # This is mainly used for betas paramaeter which is a tuple like : [[0, 0.99], [0.9, 0.99]]
                    optim_params[ele][key] = grid_params[key]
with open('../optim_params.json', 'w+') as file:
    json.dump(optim_params, file, indent=4)