'''
Run transfer analysis pipeline

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
01/08/2021

Runs analysis of transferability to new participants
For each participant, prepares all other data as train data and own data as test data
Summarises & saves results
'''
# Imports
from datahandling import *
from resultshandling import *
from run_linear import *
from run_ae import *
from analysis_group import *
from plotting import *

# Define analyses to run
def run_analysis_transfer(participants: list, flags: dict):
    '''
    Runs analysis of transferability
    Takes list of participant dicts as input
    Also takes flags dict: determines behaviour of analysis script
    
    Trains models on all other participants' data and tests performance
    '''
    
    # Create lists to keep track of all individual results
    all_results = []
    
    # Cycle through participants
    for p_id, participant in enumerate(participants):
        # Display current participant
        print(f"Transfer: participant {p_id+1}/{len(participants)} {((p_id+1)/len(participants))*100:.2f}%")
        
        # Get data
        data = get_data_transfer(participants, participant, flags)
        
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
        save_results_report(results, f"{participant['subjID']}/transfer")
        save_results_file(results, f"{participant['subjID']}_transfer")
        plotting_individual(results, f"{participant['subjID']}/transfer")
        
        # Keep track of results
        all_results.append(results)
        
    # Summarise results
    group_results = group_analysis(all_results)
    
    # Save results
    save_results_report_group(group_results, 'transfer')
    save_results_file(group_results, 'transfer')
    plotting_group(group_results, 'transfer')