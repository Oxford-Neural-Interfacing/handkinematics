'''
Data handling routines for hand biomechanics analysis

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
02/08/2021

Includes routines for processing data from raw data files, saving processed data to files & loading processed data from files

By setting flags passed to functions, can skip the need to repeat pre-processing each time by loading previously processed data from disk
'''
# Import
from preprocessing import *

import pickle

# Save data structure to file
def save_obj(data: dict, filename:str):
    '''
    Saves data structure to file
    Args:
        data (dict): data structure to save
        filename (str): file to save to
    
    Allows saving of preprocessed data to avoid running time-consuming preprocessing multiple times
    '''
    # Save data to file
    with open(f"../data/obj/{filename}.pkl", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
# Load data structure from file
def load_obj(filename:str) -> dict:
    '''
    Loads data from file
    Args:
        filename (str): file to load
        
    Returns data structure
    Allows loading already processed data for analysis
    '''
    # Load data
    with open(f"../data/obj/{filename}.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Return data structure
    return data
    
# Get data for individual analyses
def get_data_individual(participant: dict, flags: dict) -> dict:
    '''
    Prepares data for an individual participant
    Args:
        participant (dict): participant data structure
        flags (dict): flags to determine behaviour
            load_from_disk: if True, loads already processed data from disk & returns data
            save_data_to_disk: if True, saves processed data structure to disk for later loading
            
    Returns data dict for use in analyses
    '''
    # If data already processed, load from disk
    if flags['load_from_disk']:
        data = load_obj(f"individual_{participant['subjID']}")
        return data
    
    # Otherwise load & process raw data 
    # Load data
    X, Y = load_data(participant['f_id'])
    
    # Prepare data
    data = split_data(X, Y)
    data = rescale_data(**data)
    
    # If required, save data to disk
    if flags['save_data_to_disk']:
        save_obj(data, f"individual_{participant['subjID']}")
    
    # Return processed data
    return data

# Get data for grand-average analyses
def get_data_grandaverage(participants:list, flags:dict) -> dict:
    '''
    Prepares data for grand-average analyses
    Args:
        participants (list): list of participant data structures
        flags (dict): flags to determine behaviour
            load_from_disk: if True, loads already processed data from disk & returns data
            save_data_to_disk: if True, saves processed data structure to disk for later loading
            
    Returns data dict for use in analyses
    '''
    # If data already processed, load from disk
    if flags['load_from_disk']:
        data = load_obj("grandaverage")
        return data
    
    # Otherwise load & process raw data
    # Load & prepare data: all participants
    X, Y = load_all_data(participants)
    
    # Split & rescale data
    data = split_data(X, Y)
    data = rescale_data(**data)
    
    # If required, save data to disk
    if flags['save_data_to_disk']:
        save_obj(data, "grandaverage")
        
    # Return processed data
    return data

# Get data for transfer analyses
def get_data_transfer(participants: list, exclude:dict, flags:dict) -> dict:
    '''
    Prepares data for transfer analyses
    Args:
        participants (list): list of participant data structures
        exclude (dict): participant structure to exclude as test data
        flags (dict): flags to determine behaviour
            load_from_disk: if True, loads already processed data from disk & returns data
            save_data_to_disk: if True, saves processed data structure to disk for later loading
            
    Returns data dict for use in analyses
    '''
    # If data already processed, load from disk
    if flags['load_from_disk']:
        data = load_obj(f"transfer_{exclude['subjID']}")
        return data
    
    # Otherwise load & process raw data
    # Prepare train data: all other participants
    X_others, Y_others = load_all_data(participants, exclude=exclude)
    data_others = split_data(X_others, Y_others)
        
    # Prepare test data: own data
    X_participant, Y_participant = load_data(exclude['f_id'])
        
    # Prepare final data structure
    data = {
        'X_train': np.concatenate((data_others['X_train'], data_others['X_test']), axis=0), # 80% of other data
        'X_test': data_others['X_validate'], # 20% of other data
        'X_validate': X_participant, # Own data
        'Y_train': np.concatenate((data_others['Y_train'], data_others['Y_test']), axis=0),
        'Y_test': data_others['Y_validate'],
        'Y_validate': Y_participant
    }
        
    # Rescale data
    data = rescale_data(**data)
    
    # If required, save data to disk
    if flags['save_data_to_disk']:
        save_obj(data, f"transfer_{exclude['subjID']}")
        
    # Return processed data
    return data