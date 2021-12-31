'''
Run postprocessing for hand biomechanics data

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
03/09/2021

Produces final reports & figures
Loads all results files & produces group results
'''
# Imports
import pickle

from analysis_group import *
from resultshandling import *
from plotting import *
from runANOVAs import *

# Define routine for loading results
def getResults(participant:str) -> dict:
    '''
    Loads all results for a given participant
    
    Args:
        participant (str): ID for participant
    
    Returns results dict containing all results for that participant
    '''
    # Get individual results
    with open(f'../results/obj/{participant}_individual.pkl', 'rb') as f:
        individual_results = pickle.load(f)
        
    # Get transfer results
    with open(f'../results/obj/{participant}_transfer.pkl', 'rb') as f:
        transfer_results = pickle.load(f)
    
    # Return results
    return individual_results, transfer_results

# List IDs of all participants to include
participants = [
    'ck',
    'gm',
    'lc',
    'sc',
    'chp',
    'cp',
    'je',
    'jof',
    'nl',
    'sm',
    'ad',
    'az',
    'jf',
    'lic',
    'ra'
]

# Initialise list of results
individual_results = []
transfer_results = []

# Loop through all participants
for participant in participants:
    # Get results for this participant
    participant_results_individual, participant_results_transfer = getResults(participant)
    
    # Add individual trial number to transfer results
    participant_results_transfer['num_trials'] = participant_results_individual['num_trials']
    
    # Get individual report for this participant
    #save_results_report(participant_results_individual, f'{participant}/individual')
    #save_results_report(participant_results_transfer, f'{participant}/transfer')
    
    # Get individual plots for this participant
    #plotting_individual(participant_results_individual, f'{participant}/individual')
    #plotting_individual(participant_results_transfer, f'{participant}/transfer')
    
    # Save results for group analysis
    individual_results.append(participant_results_individual)
    transfer_results.append(participant_results_transfer)
    
# Get group report for all participants
group_results_individual = group_analysis(individual_results)
group_results_transfer = group_analysis(transfer_results)

save_results_report_group(group_results_individual, 'group')
save_results_report_group(group_results_transfer, 'transfer')

# Get final plots for all participants
#plotting_group(group_results_individual, 'group')
#plotting_group(group_results_transfer, 'transfer')

# Run ANOVAs (& print results)
df = run_ANOVA(group_results_individual, group_results_transfer)

# Do grandaverage analysis
with open('../results/obj/grandaverage.pkl', 'rb') as f:
    grandaverage_results = pickle.load(f)
    
# Add placeholder num_trials
grandaverage_results['num_trials'] = 'All participants combined'
    
# Get overall PCA plot
#overall_pca(grandaverage_results)
    # Edit function: save PCA model & variance explained by AE and use this
    
# Get results report
save_results_report(grandaverage_results, 'grandaverage/grandaverage')

# Get grandaverage plots
plotting_individual(grandaverage_results, 'grandaverage/grandaverage')

# Save processed results
all_results = {
    'individual': group_results_individual,
    'transfer': group_results_transfer,
    'grandaverage': grandaverage_results
}
save_results_file(all_results, 'processed_results')