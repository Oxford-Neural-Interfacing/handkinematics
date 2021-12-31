'''
Evaluate VAE on provided data

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
01/08/2021

Defines analysis pipeline for assessing VAE methods
'''
# Imports
import numpy as np

from routines_ae import *

# Define AE analyses to run
def run_ae(data:dict, flags:dict) -> dict:
    '''
    Evaluates performance of AE for reconstruction, classification and interpolation using provided data
    
    Args:
        data (dict): train, test & validate splits for assessing data
        flags (dict): flags for determining behaviour of analysis
        
    Returns:
        results (dict): results of analyses
    '''
    # Prepare additional arguments
    args = {
        'num_dims': 9, # Number of dimensions to use; ?determine from overall data
        'task_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # List of task IDs
    }
    args.update({'is_transfer': flags['is_transfer']})
    
    # Assess reconstruction accuracy
    print("\t\tAssessing reconstruction accuracy...")
    reconstruction_results = reconstruct_ae(data, args)
    
    # Assess classification accuracy
    print("\t\tAssessing classification performance...")
    classification_results = classify_ae(data, args)
    
    # Assess interpolation accuracy
    print("\t\tAssessing interpolation accuracy...")
    interpolation_results = interpolate_ae(data, args)
    
    # Define results to return
    results = {
        **reconstruction_results,
        **classification_results,
        **interpolation_results
    }
    return results
