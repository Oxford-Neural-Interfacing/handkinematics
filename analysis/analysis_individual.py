'''
Run individual analysis pipeline

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
01/08/2021

Runs analyses for a single individual
Prepares train/test data then runs linear & DL-based analyses
Saves results to individual folder & to participant structure for group analysis
'''
# Imports
from datahandling import *
from resultshandling import *
from run_linear import *
from run_ae import *
from plotting import *

# Define analyses to run
def run_analysis_individual(participant: dict, flags: dict):
    '''
    Runs analysis for a single participant
    Takes participant dict as input
    Also takes flags dict: determines behaviour of analysis
    
    Runs preprocessing then runs linear & DL-based analyses
    '''
    # Get data
    data = get_data_individual(participant, flags)

    # Run linear analyses
    print("\tRunning linear analyses...")
    results_linear = run_linear(data, flags)
    
    # Run DL analyses
    print("\tRunning nonlinear analyses...")
    results_ae = run_ae(data, flags)
    
    # Save results
    participant.update({
        'results': {
            'num_trials': data['X_train'].shape[0] + data['X_test'].shape[0] + data['X_validate'].shape[0],
            **results_linear,
            **results_ae
        }
    })
    save_results_report(participant['results'], f"{participant['subjID']}/individual")
    save_results_file(participant['results'], f"{participant['subjID']}_individual")
    plotting_individual(participant['results'], f"{participant['subjID']}/individual")