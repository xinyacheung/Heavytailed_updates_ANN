# update_tracker.py
import torch
import numpy as np
from collections import defaultdict

class CNN10UpdateTracker:
    def __init__(self, model):
        self.model = model
        self.updates_record = {
            'conv1': {'weight': [], 'bias': []},
            'conv2': {'weight': [], 'bias': []},
            'fc1': {'weight': [], 'bias': []},
            'fc2': {'weight': [], 'bias': []}
        }
        self.mag_record = {
            'conv1': {'weight': [], 'bias': []},
            'conv2': {'weight': [], 'bias': []},
            'fc1': {'weight': [], 'bias': []},
            'fc2': {'weight': [], 'bias': []}
        }
        self.previous_params = {}
        
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name in self.updates_record:
                self.previous_params[name] = param.clone().detach()

    def track_update_magnitude(self, step):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                layer_name = name.split('.')[0]
                if layer_name not in self.updates_record:
                    continue
                    
                param_type = name.split('.')[1]  # weight or bias
                
                update = param.clone().detach() - self.previous_params[name]
                magnitude = param.clone().detach()

                if param_type == 'weight':
                    update = update.flatten()
                    update_magnitude = update.abs().cpu().numpy() 
                    magnitude = magnitude.flatten().cpu().numpy()

                elif param_type == 'bias':
                    update_magnitude = update.abs().cpu().numpy()
                    magnitude = magnitude.cpu().numpy()
                
                self.updates_record[layer_name][param_type].append(update_magnitude)
                self.previous_params[name] = param.clone().detach()

                self.mag_record[layer_name][param_type].append(magnitude)


    def save_detailed_updates(self, save_path_prefix):
        for layer_name in ['conv1', 'conv2', 'fc1', 'fc2']:
            weight_updates = np.vstack(np.array(self.updates_record[layer_name]['weight']))
            np.save(f"{save_path_prefix}_{layer_name}_weights.npy", weight_updates)

            weight_mags = np.vstack(np.array(self.mag_record[layer_name]['weight']))
            np.save(f"{save_path_prefix}_{layer_name}_weight_mag.npy", weight_mags)

        for layer_name in ['conv1', 'conv2', 'fc1', 'fc2']:
            bias_updates = np.vstack(np.array(self.updates_record[layer_name]['bias']))
            np.save(f"{save_path_prefix}_{layer_name}_biases.npy", bias_updates)

            bias_mags = np.vstack(np.array(self.mag_record[layer_name]['bias']))
            np.save(f"{save_path_prefix}_{layer_name}_bias_mag.npy", bias_mags)

