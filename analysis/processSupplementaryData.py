'''
Process & save supplementary data

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
11/11/2021

Loads supplementary data files for visualisation
Performs processing
Prepares data structure
Saves results file
'''
# Imports
import numpy as np

from resultshandling import *

# Utility functions
def findMatches(X: np.ndarray, Y: np.ndarray) -> list:
    '''
    Find indices of Y that correspond to elements of X
    Used for getting matching sets of data for interpolation / transfer comparisons
    Args:
        X: smaller set of actual data
        Y: data of which X is a subset
    Returns:
        inds: list of indices of Y that map to values of X
            i.e. Y[inds] = X
            Can then use inds to get subset of reconstructed data for plotting
    '''
    # Get number of trials to iterate over
    num_trials = X.shape[0]
    num_candidates = Y.shape[0]
    
    # Find element of Y with lowest L2 norm (X-Y) - i.e. closest match
    inds = np.zeros(num_trials)
    for n in range(num_trials):
        dist = np.zeros(num_candidates)
        for idx in range(num_candidates):
            dist[idx] = np.linalg.norm(X[n, :, :] - Y[idx, :, :])
            
        # Best match is element with lowest norm
        inds[n] = np.argmin(dist)
        
    # Convert to integers for indexing
    inds = inds.astype('int')
    return inds


def rescale(actual, reconstructed):
    '''
    Rescales actual & reconstructed data to range 0-1
    Used to rescale transfer data to correct scale for plotting
    Args:
        actual: original data, used to get transformation
        reconstructed: reconstructed data; transformation applied to both
    Returns:
        actual, reconstructed: rescaled versions of original arrays
    '''
    # Reshape arrays
    n_trials, n_t, n_channels = actual.shape
    actual = np.reshape(actual, (n_trials*n_t, n_channels))
    reconstructed = np.reshape(reconstructed, (n_trials*n_t, n_channels))
    
    # Rescale data
    scaler = MinMaxScaler()
    scaler.fit(actual)
    actual = scaler.transform(actual)
    reconstructed = scaler.transform(reconstructed)
    
    # Reshape to original shape
    actual = np.reshape(actual, (n_trials, n_t, n_channels))
    reconstructed = np.reshape (reconstructed, (n_trials, n_t, n_channels))
    
    # Return rescaled data
    return actual, reconstructed

# Load data
pca = load_results('supplementary/grandaverage_pca_model')

individual = load_results('supplementary/ck_individual_example')

transfer_linear_direct = load_results('supplementary/ck_transfer_linear_reconstruction_example')
transfer_ae_direct = load_results('supplementary/ck_transfer_ae_reconstruction_example')
transfer_linear_interpolated = load_results('supplementary/ck_transfer_linear_interpolation_example')
transfer_ae_interpolated = load_results('supplementary/ck_transfer_ae_interpolation_example')

# Process data

### PCA ###
# Extract PCA model
pca = pca['pca']

# Get PCA weights
pca_weights = pca.components_

### Examples: individual, direct ###
actual = individual['actual_linear']
actual_individual_ae_direct = individual['actual_ae']

# Get data
individual_linear_direct = individual['reconstructed_linear']
individual_ae_direct = individual['reconstructed_ae']

# Reshape actual AE data
actual_individual_ae_direct = np.reshape(actual_individual_ae_direct, (individual_ae_direct.shape))

# Get matching trials (AE only, linear is reference for all others)
inds = findMatches(actual, actual_individual_ae_direct)
actual_individual_ae_direct = actual_individual_ae_direct[inds]
individual_ae_direct = individual_ae_direct[inds]

# Rescale to range 0 - 1
actual, individual_linear_direct = rescale(actual, individual_linear_direct)
actual_individual_ae_direct, individual_ae_direct = rescale(actual_individual_ae_direct, individual_ae_direct)

### Examples: individual, interpolated ###
# Get data
actual_individual_linear_interpolated = individual['linear_actual_interpolated']
individual_linear_interpolated = individual['linear_reconstructed_interpolated']

actual_individual_ae_interpolated = individual['ae_actual_interpolated']
individual_ae_interpolated = individual['ae_reconstructed_interpolated']

# Identify best-matched trials
inds = findMatches(actual, actual_individual_linear_interpolated)
actual_individual_linear_interpolated = actual_individual_linear_interpolated[inds]
individual_linear_interpolated = individual_linear_interpolated[inds]

inds = findMatches(actual, actual_individual_ae_interpolated)
actual_individual_ae_interpolated = actual_individual_ae_interpolated[inds]
individual_ae_interpolated = individual_ae_interpolated[inds]

# Rescale
actual_individual_linear_interpolated, individual_linear_interpolated = rescale(actual_individual_linear_interpolated, individual_linear_interpolated)

actual_individual_ae_interpolated, individual_ae_interpolated = rescale(actual_individual_ae_interpolated, individual_ae_interpolated)

### Examples: transfer, direct ###
# Get data
actual_transfer_linear_direct = transfer_linear_direct['actual_linear']
transfer_linear_direct = transfer_linear_direct['reconstructed_linear']

actual_transfer_ae_direct = transfer_ae_direct['actual_ae']
transfer_ae_direct = transfer_ae_direct['reconstructed_ae']

# Reshape actual AE data
actual_transfer_ae_direct = np.reshape(actual_transfer_ae_direct, (transfer_ae_direct.shape))

# Find best-matched trials
inds = findMatches(actual, actual_transfer_linear_direct)
actual_transfer_linear_direct = actual_transfer_linear_direct[inds]
transfer_linear_direct = transfer_linear_direct[inds]

inds = findMatches(actual, actual_transfer_ae_direct)
actual_transfer_ae_direct = actual_transfer_ae_direct[inds]
transfer_ae_direct = transfer_ae_direct[inds]

# Rescale
actual_transfer_linear_direct, transfer_linear_direct = rescale(actual_transfer_linear_direct, transfer_linear_direct)

actual_transfer_ae_direct, transfer_ae_direct = rescale(actual_transfer_ae_direct, transfer_ae_direct)

### Examples: transfer, interpolated ###
# Get data
actual_transfer_linear_interpolated = transfer_linear_interpolated['linear_actual_interpolated']
transfer_linear_interpolated = transfer_linear_interpolated['linear_reconstructed_interpolated']

actual_transfer_ae_interpolated = transfer_ae_interpolated['ae_actual_interpolated']
transfer_ae_interpolated = transfer_ae_interpolated['ae_reconstructed_interpolated']

# Find best-matched trials
inds = findMatches(actual, actual_transfer_linear_interpolated)
actual_transfer_linear_interpolated = actual_transfer_linear_interpolated[inds]
transfer_linear_interpolated = transfer_linear_interpolated[inds]

inds = findMatches(actual, actual_transfer_ae_interpolated)
actual_transfer_ae_interpolated = actual_transfer_ae_interpolated[inds]
transfer_ae_interpolated = transfer_ae_interpolated[inds]

# Rescale
actual_transfer_linear_interpolated, transfer_linear_interpolated = rescale(actual_transfer_linear_interpolated, transfer_linear_interpolated)

actual_transfer_ae_interpolated, transfer_ae_interpolated = rescale(actual_transfer_ae_interpolated, transfer_ae_interpolated)

# Save data structure
supplementary_results = {
    'pca': pca,
    'pca_weights': pca_weights,
    
    'individual_linear_direct': individual_linear_direct,
    'individual_linear_interpolated': individual_linear_interpolated,
    'individual_ae_direct': individual_ae_direct,
    'individual_ae_interpolated': individual_ae_interpolated,
    
    'transfer_linear_direct': transfer_linear_direct,
    'transfer_linear_interpolated': transfer_linear_interpolated,
    'transfer_ae_direct': transfer_ae_direct,
    'transfer_ae_interpolated': transfer_ae_interpolated,
    
    'actual': actual
}
save_results_file(supplementary_results, 'supplementary_results')
    # Load data in NB and identify trials to use
        # Run through each & ensure properly matched trials & scaling
    # Full plot script