'''
Run grand-average analysis pipeline

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
01/08/2021

Runs analyses using data for all participants
Prepares group train/test data then runs linear & DL-based analyses
Saves results
'''
# Imports
from datahandling import *
from resultshandling import *
from run_linear import *
from run_ae import *
from plotting import *

# Define analyses to run
def run_analysis_grandaverage(participants: list, flags: dict):
    '''
    Runs grand-average analysis for all participants
    Takes list of participant dicts as input
    Also takes flags dict: determines behaviour of analysis script
    
    Prepares data from all participants & runs analyses
    '''
    # Get data
    data = get_data_grandaverage(participants, flags)
    
    # Get overall PCA plot
    overall_pca(data)
    
    # Run linear analyses
    print("\tRunning linear analyses...")
    results_linear = run_linear(data, flags)
    
    # Run DL analyses
    print("\tRunning nonlinear analyses...")
    results_ae = run_ae(data, flags)
    
    # Save results
    results = {
        'num_trials': data['X_train'].shape[0] + data['X_test'].shape[0] + data['X_validate'].shape[0],
        **results_linear,
        **results_ae
    }
    save_results_report(results, f"grandaverage")
    save_results_file(results, f"grandaverage")
    plotting_individual(results, "grandaverage")