'''
Preprocessing routines for hand biomechanics analysis

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
02/08/2021

Includes routines for loading & preparing data, train/test splitting and rescaling
'''
# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

# Define mapping taskID -> numbers
task_ID = {
    'Thumbs up': 1,
    'Thumbs down': 2,
    'Spread fingers': 3,
    'Supinate': 4,
    'Supinate & spread': 5,
    'Wave': 6,
    'Point': 7,
    'Pincer grip': 8,
    'Pinch grip': 9,
    'Power grip': 10,
    'Hook grip': 11,
    'Disc grip': 12,
    'Span grip': 13,
    'Lateral grip': 14
}

# Define sensors to include
sensors = [
    'E65', # Goniometer 1, channel 1
    'E66', # Goniometer 1, channel 2
    'E67', # Goniometer 2, channel 1
    'E68', # Goniometer 2, channel 2
    'E70', # Accelerometer, x
    'E71', # Accelerometer, y
    'E72', # Accelerometer, z
    'K1',
    'K2',
    'K3',
    'K4',
    'K5',
    'K6',
    'K7',
    'K8',
    'K9',
    'K10',
    'K11',
    'K12',
    'K13',
    'K14'
]

# Get task name from ID number
def taskname(taskID: int) -> str:
    '''
    Gets task name for supplied ID number
    Returns name of task
    '''
    for task, ID in task_ID.items():
        if taskID == ID:
            return task
        
# Individual preprocessing
def individual_preprocessing(df:pd.DataFrame, filename:str) -> pd.DataFrame:
    '''
    Runs individual preprocessing steps
    Args:
        df (pd.DataFrame): dataframe of data loaded from CSV
        filename: name of file
        
    Some participants missed multiple trials due to sensor issues etc.
    Removes these trials
    '''
    # If SM: remove trials 580-675 (monitor issue)
    if 'sm_data' in filename:
        # Identify trials to remove
        remove_trials = np.arange(580, 676)
        
        # Create mask of trials to remove
        mask = df.TaskNum.isin(remove_trials)
        
        # Remove trials
        df = df[~mask]
        
    # If JE: remove 570 - 590 (monitor issue)
    if 'je_data' in filename:
        remove_trials = np.arange(570, 591)
        mask = df.TaskNum.isin(remove_trials)
        df = df[~mask]
        
    return df
    
# Routine for loading data to required format
def load_data(filename: str) -> np.ndarray:
    '''
    Loads data from file & prepares for further processing
    Takes filename as input
    Returns data as ndarray
    
    Performs initial processing to convert data to format required for analysis
    '''
    # Load file
    df = pd.read_csv(filename, header=4)
    
    # Remove whitespace from column headers
    df.columns = df.columns.str.strip()
    
    # Rename columns with spaces
    df.rename(columns={
        'Task ID': 'TaskID',
        'Task number': 'TaskNum'
    }, inplace=True)
    
    # Downsample to 128Hz
    df = df[0:-1:8]
   
    # Remove spaces from task IDs
    for idx, task in enumerate(df.TaskID.values):
        df.TaskID.values[idx] = task.strip()
        
    # Map tasks to numbers
    for task, ID in task_ID.items():
        df = df.replace(task, ID)
       
    # Do individual preprocessing - remove trials with equipment issues etc.
    df = individual_preprocessing(df, filename)
    
    # Prepare arrays
    num_sensors = len(sensors)
    num_trials = len(df.TaskNum.unique())
    num_t = 256 # 2 seconds @ 128Hz (downsampled above)
    X = np.zeros((num_trials, num_t, num_sensors)) # Time series at each sensor - trials x time x sensors
    Y = np.zeros(num_trials) # Label for each trial
    
    # Replace short trials with mean of all other trials
    short_trials = []
    for trial_idx, trial in enumerate(df.TaskNum.unique()):
        # Loop over sensors to include
        for sensor_id, sensor in enumerate(sensors):
            data_ = df[sensor][df.TaskNum==trial].values
            
            # Save trial ID
            Y[trial_idx] = df[df.TaskNum==trial].TaskID.unique()
            
            # Check length of data; cut to 256 or record short trial
            if len(data_) >= 256:
                data_ = data_[:256]
                
                # Save data
                X[trial_idx, :, sensor_id] = data_
                
            elif len(data_) < 256:
                if trial not in short_trials:
                    short_trials.append(trial)
                continue
                
    # Get indices of acceptable trials
    ok_trials = [i for i in range(num_trials) if i not in short_trials]

    # Keep only acceptable trials
    X = X[ok_trials, :, :]
    Y = Y[ok_trials]
    
    # Return data & labels
    return X, Y

# Load & combine data for multiple subjects
def load_all_data(participants:list, exclude=None) -> np.ndarray:
    '''
    Loads data from all participants; can exclude subjects if required
    
    Args:
        participants (list): list of participant dicts
        exclude: participant to exclude, if any; default None, otherwise participant dict of participant to exclude
        
    Returns data as ndarrays
    '''
    # Cycle through participants
    array_initialised = False # State variable for loop
    for participant in participants:
        # If participant excluded, skip to next
        if exclude is not None and participant['subjID'] == exclude['subjID']:
            continue
        
        # Load participant's data
        X_participant, Y_participant = load_data(participant['f_id'])
        
        # If results array not initialised, initialise it
        if array_initialised == False:
            X = X_participant
            Y = Y_participant
            array_initialised = True
            continue
        
        # Add new data to previous data
        X = np.concatenate((X, X_participant), axis=0)
        Y = np.concatenate((Y, Y_participant), axis=0)
        
    # Return combined data
    return X, Y

# Routine for splitting data into train / test / validation sets
def split_data(X:np.ndarray, Y:np.ndarray) -> dict:
    '''
    Splits data into train / test / validation sets
    Args:
        X (ndarray): time series of sensor data - trials x time x sensors
        Y (ndarray): movement classification for each trial
        
    Returns train, test and validation splits in dict
    Splits data into 60% train, 20% test, 20% validate
    '''
    # Get total number of trials
    num_trials = X.shape[0]
    
    # Get target sizes
    train_size = int(num_trials * 0.6)
    test_size = int(num_trials * 0.2)
    validation_size = int(num_trials * 0.2)
    
    # Set random seeds for generating splits
    random_seed_test = 17
    random_seed_validate = 23
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed_test)
    X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=validation_size, random_state=random_seed_validate)
   
    # Prepare data to return
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'X_validate': X_validate,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'Y_validate': Y_validate
    }
    
    # Return split data
    return data

# Rescale all data to standard scale
def rescale_data(X_train:np.ndarray, X_test:np.ndarray, X_validate:np.ndarray, Y_train:np.ndarray, Y_test:np.ndarray, Y_validate:np.ndarray) -> dict:
    '''
    Rescales all data to standard scale
    Args:
        X_* (ndarray): train/test/validation splits; time series of sensor readings - trials x time x sensors
        Y_* (ndarray): train/test/validation splits; labels for each trial
        
    Returns: rescaled train/test/validation splits as dict containing data
    '''
    # Get sizes of arrays
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    validation_size = X_validate.shape[0]
    num_t = X_train.shape[1]
    num_sensors = X_train.shape[2]
    
    # Reshape for rescaling
    X_train = np.reshape(X_train, (train_size*num_t, num_sensors))
    X_test = np.reshape(X_test, (test_size*num_t, num_sensors))
    X_validate = np.reshape(X_validate, (validation_size*num_t, num_sensors))
    
    # Rescale features - fit scaler to training data only
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    
    # Transform all data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_validate = scaler.transform(X_validate)
    
    # Reshape back to trials x time x sensors
    X_train = np.reshape(X_train, (train_size, num_t, num_sensors))
    X_test = np.reshape(X_test, (test_size, num_t, num_sensors))
    X_validate = np.reshape(X_validate, (validation_size, num_t, num_sensors))
    
    # Prepare data to return
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'X_validate': X_validate,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'Y_validate': Y_validate
    }
    
    # Return rescaled data
    return data
    
# Dataset class for returning batches of data for autoencoder training
class HandData(Dataset):
    '''
    Defines torch dataset for hand kinematic data
    Shape: trials x time x sensors
    '''
    # Define dataset constructor
    def __init__(self, x_data, y_data):
        '''
        Takes data in
        Saves as dataset attributes
        '''
        # Convert data to tensors
        self.x = torch.tensor(x_data)
        self.y = torch.tensor(y_data)
        
    # Define method for returning length of dataset
    def __len__(self):
        '''
        Called when len(dataset) used
        Returns number of samples in dataset(i.e. number of trials)
        '''
        return self.x.shape[0] # Returns number of trials
    
    # Define method for getting items from dataset
    def __getitem__(self, idx):
        '''
        Called by dataloader during training/testing
        Returns batches of data with corresponding labels
        '''
        # If multiple indices requested, convert to list for indexing
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get desired trials
        x_data = self.x[idx, :, :]
        y_data = self.y[idx]
        
        return x_data, y_data