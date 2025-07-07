# update_tracker.py
import torch
import numpy as np
from collections import defaultdict

class MLPUpdateTracker:
    def __init__(self, model):
        self.model = model
        self.updates_record = {
            'fc1': {'weight': [], 'bias': []},
            'fc2': {'weight': [], 'bias': []}
        }
        self.mag_record = {
            'fc1': {'weight': [], 'bias': []},
            'fc2': {'weight': [], 'bias': []}
        }
        self.previous_params = {}
        
        for name, param in model.named_parameters():
            if 'fc' in name:
                self.previous_params[name] = param.clone().detach()

    def track_update_magnitude(self, step):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    continue
                    
                layer_name = name.split('.')[0]  # fc1 or fc2
                param_type = name.split('.')[1]  # weight or bias
                
                update = param.clone().detach() - self.previous_params[name]
                magnitude = param.clone().detach()

                if param_type == 'weight':
                    update = update.flatten()
                    update_magnitude = update.abs().cpu().numpy()  # Per output unit (neuron)
                    magnitude = magnitude.flatten().cpu().numpy()

                elif param_type == 'bias':
                    update_magnitude = update.abs().cpu().numpy()
                    magnitude = magnitude.cpu().numpy()

                self.updates_record[layer_name][param_type].append(update_magnitude)
                self.mag_record[layer_name][param_type].append(magnitude)

                self.previous_params[name] = param.clone().detach()

    def save_updates(self, save_path):

        fc1_weight = np.array(self.updates_record['fc1']['weight'])
        fc1_bias = np.array(self.updates_record['fc1']['bias'])
        fc2_weight = np.array(self.updates_record['fc2']['weight'])
        fc2_bias = np.array(self.updates_record['fc2']['bias'])
        

        linear_updates = np.vstack([
            fc1_weight, 
            fc2_weight   
        ])
        
        np.save(save_path, linear_updates)

    def save_detailed_updates(self, save_path_prefix):
        for layer_name in ['fc1', 'fc2']:
            weight_updates = np.vstack(np.array(self.updates_record[layer_name]['weight']))
            np.save(f"{save_path_prefix}_{layer_name}_weights.npy", weight_updates)
            weight_updates = np.vstack(np.array(self.mag_record[layer_name]['weight']))
            np.save(f"{save_path_prefix}_{layer_name}_weights_mag.npy", weight_updates)

        for layer_name in ['fc1', 'fc2']:
            bias_updates = np.vstack(np.array(self.updates_record[layer_name]['bias']))
            np.save(f"{save_path_prefix}_{layer_name}_biases.npy", bias_updates)
            bias_updates = np.vstack(np.array(self.mag_record[layer_name]['bias']))
            np.save(f"{save_path_prefix}_{layer_name}_biases_mag.npy", bias_updates)
            