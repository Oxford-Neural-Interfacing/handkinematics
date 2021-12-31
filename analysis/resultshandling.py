'''
Results handling routines for hand biomechanics analysis

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
04/08/2021

Includes routines for saving results to text reports & to files to allow re-loading and postprocessing results
'''
# Imports
import pickle
from preprocessing import *

# Save results to text report
def save_results_report(results: dict, filename: str):
    '''
    Save results structure to text report
    Args:
        results (dict): results data structure
        filename (str): file to save to
        
    Creates text summary of analysis report for easy review
    '''
    with open(f"../results/{filename}_results.txt", "w") as f:
        f.write(f"""Results summary:
        Trials: {results['num_trials']}
        
        Reconstruction:
            Linear reconstruction MSE: {results['mse_linear']}
            DL reconstruction MSE: {results['mse_ae']}
            
        Classification:
            Maximum classification accuracy (linear): {results['accuracy_linear_max']}
            Baseline (linear): {results['accuracy_baseline_linear_max']}

            Maximum classification accuracy (DL): {results['accuracy_ae_max']}
            Baseline (DL): {results['accuracy_baseline_ae_max']}
            
        Interpolation:
            Mean interpolation loss (linear): {results['linear_interpolation_loss_mean']}
            Ranked tasks (linear):
""")
        
        for idx, task in enumerate(results['linear_ranked_tasks']):
            f.write(f"\t\t{taskname(task)}: {results['linear_ranked_loss'][idx]}\n")
            
        f.write(f"""
            Mean interpolation loss (DL): {results['ae_interpolation_loss_mean']}
            Ranked tasks (DL):
""")
        for idx, task in enumerate(results['ae_ranked_tasks']):
            f.write(f"\t\t{taskname(task)}: {results['ae_ranked_loss'][idx]}\n")

# Save results to text report - group
def save_results_report_group(results:dict, filename:str):
    '''
    Save results structure to text report - group results
    Args:
        results (dict): results data structure
        filename (str): file to save to
        
    Creates text summary of analysis report for easy review
    '''
    with open(f"../results/{filename}_results.txt", "w") as f:
        f.write(f"""Results summary:
        Reconstruction:
            Linear reconstruction MSE: {results['mse_linear_mean']} +/- {results['mse_linear_std']}
            AE reconstruction MSE: {results['mse_ae_mean']} +/- {results['mse_ae_std']}
            Paired sample t-test: p = {results['mse_p']}
            
        Classification:
            Baseline: {results['accuracy_baseline_max_mean']} +/- {results['accuracy_baseline_max_std']}
            Maximum classification accuracy (linear): {results['accuracy_linear_max_mean']} +/- {results['accuracy_linear_max_std']}
            Maximum classification accuracy (DL): {results['accuracy_ae_max_mean']} +/- {results['accuracy_ae_max_std']}
            Paired sample t-test: p = {results['accuracy_max_p']}
            
            Per classifier:
            """)
        
        for idx, classifier in enumerate(results['classifiers']):
            f.write(f""""\t\t{classifier}:
            Baseline: {results['accuracy_baseline_mean'][idx]} +/- {results['accuracy_baseline_std'][idx]}
            Linear: {results['accuracy_linear_mean'][idx]} +/- {results['accuracy_linear_std'][idx]}
            DL: {results['accuracy_ae_mean'][idx]} +/- {results['accuracy_ae_std'][idx]}
            Paired sample t-test: p = {results['accuracy_classifiers_p'][idx]}\n""")
            
        f.write(f"""
        Interpolation:
            Mean interpolation loss (linear): {results['linear_interpolation_loss_mean_mean']} +/- {results['linear_interpolation_loss_mean_std']}
            Mean interpolation loss (DL): {results['ae_interpolation_loss_mean_mean']} +/- {results['ae_interpolation_loss_mean_std']}
            Paired sample t-test: p = {results['interpolation_loss_p']}
            
            Standard vs. interpolated, PCA: p = {results['interpolation_loss_p_pca']}
            Standard vs. interpolated, AE: p = {results['interpolation_loss_p_ae']}
            
            Ranked tasks (linear):
""")
        
        for idx, task in enumerate(results['linear_ranked_tasks']):
            f.write(f"\t\t{taskname(task)}: {results['linear_ranked_loss_mean'][idx]} +/- {results['linear_ranked_loss_std'][idx]}\n")
            
        f.write(f"\nRanked tasks (DL):\n")
        for idx, task in enumerate(results['ae_ranked_tasks']):
            f.write(f"\t\t{taskname(task)}: {results['ae_ranked_loss_mean'][idx]} +/- {results['ae_ranked_loss_std'][idx]}\n")
    
# Save results to file
def save_results_file(results: dict, filename: str):
    '''
    Save results structure to file
    Args:
        results (dict): results data structure
        filename (str): file to save to
        
    Saves to file for later re-loading
    '''
    # Save results
    with open(f"../results/obj/{filename}.pkl", "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

# Load results from file
def load_results(filename: str) -> dict:
    '''
    Loads results from file
    Args:
        filename (str): file to load
        
    Returns results structure
    '''
    # Load results
    with open(f"../results/obj/{filename}.pkl", "rb") as f:
        results = pickle.load(f)
        
    # Return results structure
    return results
