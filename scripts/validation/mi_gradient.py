import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

bs=500
data_path = f'../MNIST-5mlp/bs{bs}/'

def get_weight_output(data_path, opti, lr, cate, suffixes):
    layer_list = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    unit_all_list = [] 
    for dig in range(10):  
        unit_all = []  
        for idx, suffix in enumerate(suffixes):
            st = int(suffix.split('-')[0])
            ed = int(suffix.split('-')[1])

            for steps in np.arange(st, ed+1, 5000):
                uni = np.load(data_path + f'{cate}_optim{opti}_lr{lr}_{layer_list[0]}_digit{dig}_steps{ed}.npy')
                expanded_columns = []
                for i in range(uni.shape[1]):
                    expanded_column = np.tile(uni[:, i], (784, 1)).T 
                    expanded_columns.append(expanded_column)
                uni = np.concatenate(expanded_columns, axis=1)

                for ll in layer_list[1:]:
                    temp = np.load(data_path + f'{cate}_optim{opti}_lr{lr}_{ll}_digit{dig}_steps{ed}.npy')
                    expanded_columns = []
                    for i in range(temp.shape[1]):
                        expanded_column = np.tile(temp[:, i], (50, 1)).T 
                        expanded_columns.append(expanded_column)
                    temp = np.concatenate(expanded_columns, axis=1)
                    uni = np.hstack([uni, temp])
                unit_all.append(uni)

            unit_all = np.vstack(unit_all)
            unit_all_list.append(unit_all)
    unit_all_avg = np.nanmean(unit_all_list, axis=0)

    layer_list = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']
    weight_all = []
    for steps in np.arange(st, ed+1, 5000):
        weights_updates = np.load(data_path + f'optim_{opti}_lr{lr}_steps{steps}_{layer_list[0]}_weights.npy')
        
        for ll in layer_list[1:]:
            weights_updates = np.hstack([weights_updates, np.load(data_path + f'energy_optim_{opti}_lr{lr}_steps{steps}_{ll}_weights.npy')])
        weight_all.append(weights_updates)
    weight_all = np.vstack(weight_all)

    return weight_all, unit_all_avg

def mutual_information(w, a, bins=50):
    w = (w - np.mean(w)) / (np.std(w) + 1e-10)  
    a = (a - np.mean(a)) / (np.std(a) + 1e-10) 
    
    joint_hist, _, _ = np.histogram2d(w.flatten(), a.flatten(), bins=bins)
    joint_prob = joint_hist / joint_hist.sum()
    
    w_prob = joint_prob.sum(axis=1, keepdims=True)
    a_prob = joint_prob.sum(axis=0, keepdims=True)
    
    joint_prob += 1e-10  # avoid log(0)
    w_prob += 1e-10
    a_prob += 1e-10

    mi = np.sum(joint_prob * np.log(joint_prob / (w_prob @ a_prob)))
    return mi

def PCA_direction(weight_all, step_interval=200):
    pca_cosine_dist = []
    for idx in range(0, weight_all.shape[0] - 1, step_interval):
        current_weights = weight_all[idx:idx + step_interval, :]
        print(current_weights.shape)
        pca = PCA(n_components=100) 
        pca.fit(current_weights)
        pca_direction = pca.components_[0]
        if idx > 0:
            pca_cosine_dist.append(np.abs(1 - cosine(pca_direction, prev_pca_direction)))
        prev_pca_direction = pca_direction

