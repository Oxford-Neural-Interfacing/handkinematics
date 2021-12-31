'''
Run group analysis pipeline

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
01/08/2021

Runs group-level analysis of individual results
Prepares summary statistics for comparison of methods at individual level
Saves results
'''
# Imports
import numpy as np
from scipy import stats

from resultshandling import *
from plotting import *

# Define handling of group results
def group_analysis(results:list) -> dict:
    '''
    Runs group-level analysis on individual results
    Args:
        results (list): list of results dicts
        
    Returns:
        dict: group-level results
        
    Takes list of results dicts
    Returns dict of group-level results
    Can then be used for reports & plotting
    '''
    
    # Get mean & SD reconstruction error
    mse_linear = [r['mse_linear'] for r in results]
    mse_linear_mean = np.mean(mse_linear)
    mse_linear_std = np.std(mse_linear)
    
    mse_ae = [r['mse_ae'] for r in results]
    mse_ae_mean = np.mean(mse_ae)
    mse_ae_std = np.std(mse_ae)
    
    mse_linear_dims = np.zeros((len(results), len(results[0]['mse_linear_dims'])))
    for idx, result in enumerate(results):
        mse_linear_dims[idx, :] = result['mse_linear_dims']
    mse_linear_dims_mean = np.mean(mse_linear_dims, axis=0)
    mse_linear_dims_std = np.std(mse_linear_dims, axis=0)
    
    mse_ae_dims = np.zeros((len(results), len(results[0]['mse_ae_dims'])))
    for idx, result in enumerate(results):
        mse_ae_dims[idx, :] = result['mse_ae_dims']
    mse_ae_dims_mean = np.mean(mse_ae_dims, axis=0)
    mse_ae_dims_std = np.std(mse_ae_dims, axis=0)
    
    # Compare reconstruction errors (linear vs. DL)
    mse_p = stats.ttest_rel(mse_linear, mse_ae)[1] # Paired-same t-test, MSE linear vs. AE
   
    # Accuracy vs. dimensions
    accuracy_linear_dims = np.zeros((len(results), len(results[0]['accuracy_linear_dims'])))
    for idx, result in enumerate(results):
        accuracy_linear_dims[idx, :] = result['accuracy_linear_dims']
    accuracy_linear_dims_mean = np.mean(accuracy_linear_dims, axis=0)
    accuracy_linear_dims_std = np.std(accuracy_linear_dims, axis=0)
    
    accuracy_ae_dims = np.zeros((len(results), len(results[0]['accuracy_ae_dims'])))
    for idx, result in enumerate(results):
        accuracy_ae_dims[idx, :] = result['accuracy_ae_dims']
    accuracy_ae_dims_mean = np.mean(accuracy_ae_dims, axis=0)
    accuracy_ae_dims_std = np.std(accuracy_ae_dims, axis=0)

    # Get mean & SD classification accuracy
    accuracy_linear_max = [r['accuracy_linear'] for r in results]
    accuracy_linear_max_mean = np.mean(accuracy_linear_max)
    accuracy_linear_max_std = np.std(accuracy_linear_max)
    
    accuracy_ae_max = [r['accuracy_ae'] for r in results]
    accuracy_ae_max_mean = np.mean(accuracy_ae_max)
    accuracy_ae_max_std = np.std(accuracy_ae_max)
    
    classifiers = ['LR', 'SVM', 'SGD', 'NB', 'RF']
    accuracy_linear = np.zeros((len(results), len(classifiers)))
    for idx, result in enumerate(results):
        accuracy_linear[idx, :] = result['accuracy_linear']
    accuracy_linear_mean = np.mean(accuracy_linear, axis=0)
    accuracy_linear_std = np.std(accuracy_linear, axis=0)
    
    classifiers = ['LR', 'SVM', 'SGD', 'NB', 'RF']
    accuracy_ae = np.zeros((len(results), len(classifiers)))
    for idx, result in enumerate(results):
        accuracy_ae[idx, :] = result['accuracy_ae']
    accuracy_ae_mean = np.mean(accuracy_ae, axis=0)
    accuracy_ae_std = np.std(accuracy_ae, axis=0)

    # Baseline accuracies - should be same for DL & linear
    accuracy_baseline_max = [r['accuracy_baseline_linear_max'] for r in results]
    accuracy_baseline_max_mean = np.mean(accuracy_baseline_max)
    accuracy_baseline_max_std = np.std(accuracy_baseline_max)
    
    classifiers = ['LR', 'SVM', 'SGD', 'NB', 'RF']
    accuracy_baseline = np.zeros((len(results), len(classifiers)))
    for idx, result in enumerate(results):
        accuracy_baseline[idx, :] = result['accuracy_baseline_linear']
    accuracy_baseline_mean = np.mean(accuracy_linear, axis=0)
    accuracy_baseline_std = np.std(accuracy_linear, axis=0)
        # ?statistical tests, methods vs. baseline

    # Compare classification accuracy (linear vs. DL)
    accuracy_max_p = stats.ttest_rel(accuracy_linear_max, accuracy_ae_max)[1] # Max accuracy
    
    accuracy_classifiers_p = np.zeros(len(classifiers))
    for i in range(len(classifiers)):
        accuracy_classifiers_p[i] = stats.ttest_rel(accuracy_linear[:,i], accuracy_ae[:,i])[1] # Accuracy per classifier
        
    # Get mean & SD interpolation error
    linear_interpolation_loss_mean = [r['linear_interpolation_loss_mean'] for r in results]
    linear_interpolation_loss_mean_mean = np.mean(linear_interpolation_loss_mean)
    linear_interpolation_loss_mean_std = np.std(linear_interpolation_loss_mean)
    
    ae_interpolation_loss_mean = [r['ae_interpolation_loss_mean'] for r in results]
    ae_interpolation_loss_mean_mean = np.mean(ae_interpolation_loss_mean)
    ae_interpolation_loss_mean_std = np.std(ae_interpolation_loss_mean)
    
    # Compare interpolation errors (linear vs. DL)
    interpolation_loss_p = stats.ttest_rel(linear_interpolation_loss_mean, ae_interpolation_loss_mean)[1] # Interpolation loss
    
    interpolation_loss_p_pca = stats.ttest_rel(mse_linear, linear_interpolation_loss_mean)[1]
    interpolation_loss_p_ae = stats.ttest_rel(mse_ae, ae_interpolation_loss_mean)[1]
    
    # Get overall ranked lists of movements (linear & DL)
    linear_interpolation_loss = np.zeros((len(results), len(results[0]['linear_interpolation_loss'])))
    for idx, result in enumerate(results):
        linear_interpolation_loss[idx, :] = result['linear_interpolation_loss']
    linear_interpolation_loss_all_mean = np.mean(linear_interpolation_loss, axis=0)
    linear_interpolation_loss_all_std = np.std(linear_interpolation_loss, axis=0)
    
    ae_interpolation_loss = np.zeros((len(results), len(results[0]['ae_interpolation_loss'])))
    for idx, result in enumerate(results):
        ae_interpolation_loss[idx, :] = result['ae_interpolation_loss']
    ae_interpolation_loss_all_mean = np.mean(ae_interpolation_loss, axis=0)
    ae_interpolation_loss_all_std = np.std(ae_interpolation_loss, axis=0)
    
    tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    linear_ranked_tasks = [x for _, x in sorted(zip(linear_interpolation_loss_all_mean, tasks), key=lambda pair: pair[0], reverse=True)]
    linear_ranked_loss_mean = sorted(linear_interpolation_loss_all_mean, reverse=True)
    linear_ranked_loss_std = sorted(linear_interpolation_loss_all_std, reverse=True)
    
    tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ae_ranked_tasks = [x for _, x in sorted(zip(ae_interpolation_loss_all_mean, tasks), key=lambda pair: pair[0], reverse=True)]
    ae_ranked_loss_mean = sorted(ae_interpolation_loss_all_mean, reverse=True)
    ae_ranked_loss_std = sorted(ae_interpolation_loss_all_std, reverse=True)
    
    # Get overall average impact of number of movements (linear & DL)
    linear_interpolation_loss_tasknum = np.zeros((len(results), len(results[0]['linear_interpolation_loss_tasknum'])))
    for idx, result in enumerate(results):
        linear_interpolation_loss_tasknum[idx, :] = result['linear_interpolation_loss_tasknum']
    linear_interpolation_loss_tasknum_mean = np.mean(linear_interpolation_loss_tasknum, axis=0)
    linear_interpolation_loss_tasknum_std = np.std(linear_interpolation_loss_tasknum, axis=0)
   
    ae_interpolation_loss_tasknum = np.zeros((len(results), len(results[0]['ae_interpolation_loss_tasknum'])))
    for idx, result in enumerate(results):
        ae_interpolation_loss_tasknum[idx, :] = result['ae_interpolation_loss_tasknum']
    ae_interpolation_loss_tasknum_mean = np.mean(ae_interpolation_loss_tasknum, axis=0)
    ae_interpolation_loss_tasknum_std = np.std(ae_interpolation_loss_tasknum, axis=0)
    
    linear_interpolation_accuracy_tasknum = np.zeros((len(results), len(results[0]['linear_interpolation_accuracy_tasknum'])))
    for idx, result in enumerate(results):
        linear_interpolation_accuracy_tasknum[idx, :] = result['linear_interpolation_accuracy_tasknum']
    linear_interpolation_accuracy_tasknum_mean = np.mean(linear_interpolation_accuracy_tasknum, axis=0)
    linear_interpolation_accuracy_tasknum_std = np.std(linear_interpolation_accuracy_tasknum, axis=0)
    
    ae_interpolation_accuracy_tasknum = np.zeros((len(results), len(results[0]['ae_interpolation_accuracy_tasknum'])))
    for idx, result in enumerate(results):
        ae_interpolation_accuracy_tasknum[idx, :] = result['ae_interpolation_accuracy_tasknum']
    ae_interpolation_accuracy_tasknum_mean = np.mean(ae_interpolation_accuracy_tasknum, axis=0)
    ae_interpolation_accuracy_tasknum_std = np.std(ae_interpolation_accuracy_tasknum, axis=0)
    
    # Prepare results structure
    group_results = {
        'mse_linear': mse_linear,
        'mse_linear_mean': mse_linear_mean,
        'mse_linear_std': mse_linear_std,
        
        'mse_ae': mse_ae,
        'mse_ae_mean': mse_ae_mean,
        'mse_ae_std': mse_ae_std,
        
        'mse_p': mse_p,
        
        'mse_linear_dims': mse_linear_dims,
        'mse_linear_dims_mean': mse_linear_dims_mean,
        'mse_linear_dims_std': mse_linear_dims_std,
        
        'mse_ae_dims': mse_ae_dims,
        'mse_ae_dims_mean': mse_ae_dims_mean,
        'mse_ae_dims_std': mse_ae_dims_std,

        'accuracy_linear_dims': accuracy_linear_dims,
        'accuracy_linear_dims_mean': accuracy_linear_dims_mean,
        'accuracy_linear_dims_std': accuracy_linear_dims_std,
        
        'accuracy_ae_dims': accuracy_ae_dims,
        'accuracy_ae_dims_mean': accuracy_ae_dims_mean,
        'accuracy_ae_dims_std': accuracy_ae_dims_std,
        
        'accuracy_linear_max': accuracy_linear_max,
        'accuracy_linear_max_mean': accuracy_linear_max_mean,
        'accuracy_linear_max_std': accuracy_linear_max_std,
        
        'accuracy_ae_max': accuracy_ae_max,
        'accuracy_ae_max_mean': accuracy_ae_max_mean,
        'accuracy_ae_max_std': accuracy_ae_max_std,
        
        'accuracy_max_p': accuracy_max_p,
        
        'classifiers': classifiers,
        
        'accuracy_linear': accuracy_linear,
        'accuracy_linear_mean': accuracy_linear_mean,
        'accuracy_linear_std': accuracy_linear_std,
        
        'accuracy_ae': accuracy_ae,
        'accuracy_ae_mean': accuracy_ae_mean,
        'accuracy_ae_std': accuracy_ae_std,
        
        'accuracy_classifiers_p': accuracy_classifiers_p,

        'accuracy_baseline_max': accuracy_baseline_max,
        'accuracy_baseline_max_mean': accuracy_baseline_max_mean,
        'accuracy_baseline_max_std': accuracy_baseline_max_std,

        'accuracy_baseline': accuracy_baseline,
        'accuracy_baseline_mean': accuracy_baseline_mean,
        'accuracy_baseline_std': accuracy_baseline_std,
        
        'linear_interpolation_loss_mean': linear_interpolation_loss_mean,
        'linear_interpolation_loss_mean_mean': linear_interpolation_loss_mean_mean,
        'linear_interpolation_loss_mean_std': linear_interpolation_loss_mean_std,
        
        'ae_interpolation_loss_mean': ae_interpolation_loss_mean,
        'ae_interpolation_loss_mean_mean': ae_interpolation_loss_mean_mean,
        'ae_interpolation_loss_mean_std': ae_interpolation_loss_mean_std,
        
        'interpolation_loss_p': interpolation_loss_p,
        'interpolation_loss_p_pca': interpolation_loss_p_pca,
        'interpolation_loss_p_ae': interpolation_loss_p_ae,
        
        'linear_interpolation_loss': linear_interpolation_loss,
        'linear_interpolation_loss_all_mean': linear_interpolation_loss_all_mean,
        'linear_interpolation_loss_all_std': linear_interpolation_loss_all_std,
        
        'ae_interpolation_loss': ae_interpolation_loss,
        'ae_interpolation_loss_all_mean': ae_interpolation_loss_all_mean,
        'ae_interpolation_loss_all_std': ae_interpolation_loss_all_std,
        
        'linear_ranked_tasks': linear_ranked_tasks,
        'linear_ranked_loss_mean': linear_ranked_loss_mean,
        'linear_ranked_loss_std': linear_ranked_loss_std,
        
        'ae_ranked_tasks': ae_ranked_tasks,
        'ae_ranked_loss_mean': ae_ranked_loss_mean,
        'ae_ranked_loss_std': ae_ranked_loss_std,
        
        'linear_interpolation_loss_tasknum': linear_interpolation_loss_tasknum,
        'linear_interpolation_loss_tasknum_mean': linear_interpolation_loss_tasknum_mean,
        'linear_interpolation_loss_tasknum_std': linear_interpolation_loss_tasknum_std,
        
        'ae_interpolation_loss_tasknum': ae_interpolation_loss_tasknum,
        'ae_interpolation_loss_tasknum_mean': ae_interpolation_loss_tasknum_mean,
        'ae_interpolation_loss_tasknum_std': ae_interpolation_loss_tasknum_std,
        
        'linear_interpolation_accuracy_tasknum': linear_interpolation_accuracy_tasknum,
        'linear_interpolation_accuracy_tasknum_mean': linear_interpolation_accuracy_tasknum_mean,
        'linear_interpolation_accuracy_tasknum_std':
        linear_interpolation_accuracy_tasknum_std,
        
        'ae_interpolation_accuracy_tasknum': ae_interpolation_accuracy_tasknum,
        'ae_interpolation_accuracy_tasknum_mean': ae_interpolation_accuracy_tasknum_mean,
        'ae_interpolation_accuracy_tasknum_std':
        ae_interpolation_accuracy_tasknum_std
    }
    
    # Return group-level results
    return group_results
    
# Define analyses to run
def run_analysis_group(participants:list, flags: dict):
    '''
    Runs analysis of individual-level results
    Takes list of participant dicts as input
    Also takes flags dict: determines behaviour of analysis script
    
    Produces summary statistics for comparing methods
    '''
    # Get list of results
    results = [p['results'] for p in participants]
    
    # Run group analysis
    group_results = group_analysis(results)
    
    # Save results
    save_results_report_group(group_results, 'group')
    save_results_file(group_results, 'group')
    plotting_group(group_results, 'group')
