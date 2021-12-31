'''
Network specifications for hand biomechanics analysis

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
27/08/2021

Specificies network architecture for autoencoder
Also specifies training & testing routines
'''

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from preprocessing import *

# Set manual seed
torch.manual_seed(17)

### Define model architecture ###

# Define autoencoder architecture
class AutoEncoder(nn.Module):
    '''
    Combines LSTN RNN and autoencoder
    Encoder: Input -> LSTM -> map to low-d representation
    Decoder: low-d representation -> LSTM -> map to full kinematic data
    
    Further variations could include e.g. variational component, classifier network, etc.
    '''
    def __init__(self, batch_size, input_size, hidden_size, num_dims, device):
        '''
        Model constructor
        
        Args:
            batch_size: batch size of input (number of trials)
            input_size: dimensionality of input (numer of sensors)
            hidden_size: size of hidden representation (hyperparameter)
            num_dims: number of dimensions in low-d representation (matched to PCA)
            device: device to run model on (CPU or GPU)
        '''
        # Run parent class contructor
        super(AutoEncoder, self).__init__()
        
        # Save input parameters to attributes
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_dims = num_dims
        self.device = device
        
        # Define model layers
        self.lstm_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_encoder = nn.Linear(hidden_size, num_dims)
        
        self.lstm_decoder = nn.LSTM(num_dims, hidden_size, batch_first=True)
        self.fc_decoder = nn.Linear(hidden_size, input_size)
        
        # Initialise LSTM states
        self.h_n_encoder = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        self.c_n_encoder = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        self.h_n_decoder = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        self.c_n_decoder = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        
    # Initialise hidden state
    def initHidden(self, batch_size):
        '''
        Resets LSTM state to zero
        Used at beginning of each new time series
        '''
        # Adjust batch size based on input (allows for variable batch size, e.g. last batch different size)
        self.batch_size = batch_size
        
        self.h_n_encoder = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        self.c_n_encoder = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        self.h_n_decoder = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        self.c_n_decoder = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        
    # Map inputs to low-d representation
    def encode(self, x):
        '''
        Take inputs & map to low-d representation
        Passes through LSTM & fully connected layers
        '''
        z, (self.h_n_encoder, self.c_n_encoder) = self.lstm_encoder(x, (self.h_n_encoder, self.c_n_encoder))
        z = self.fc_encoder(z.view(self.batch_size, 1, -1))
        
        return z
    
    # Map low-d representation back to full data
    def decode(self, z):
        '''
        Takes low-d representation and maps back to full data
        Passes through LSTM and fully connected layers
        '''
        y, (self.h_n_decoder, self.c_n_decoder) = self.lstm_decoder(z, (self.h_n_decoder, self.c_n_decoder))
        y = self.fc_decoder(y.view(self.batch_size, 1, -1))
        
        return y
    
    # Forward pass
    def forward(self, x):
        '''
        Takes input, encodes & decodes
        Returns reconstructed input
        '''
        # Check dimensions: if only two, add third (i.e. if single trial)
        if x.ndim == 2:
            x = x.unsqueeze(0)
            
        # Encode & decode data
        x = x.float() # Ensure data type correct
        z = self.encode(x)
        y = self.decode(z)
        
        return y
    
### Define train & test routines ####

# Define training loop
def train(args, model, loss_fn, device, train_loader, optimizer, epoch):
    '''
    Runs training loop
    
    Input parameters:
        args: dict of training parameters
            batch_size: number of trials per batch
            epochs: number of training runs
            lr: initial learning rate (**currently disabled**)
            gamma: lr step (**currently disabled**)
            no_cuda: whether to disable CUDA training
            dry_run: whether to do a single pass for testing
            seed: random seed
            log_interval: how often to log loss
            save_model: whether to save model
        model: model to train
        loss_fn: loss function to use for training
        device: device to train on
        train_loader: dataloader for training data
        optimizer: optimizer to use
        epoch: current epoch
        
    Runs training procedure using passed parameters
    Parameters can be passed as a dict using unpacking operator for convenience
    '''
    # Set model to training model
    model.train()
    
    # Log loss regularly
    epoch_loss = []
    
    # Cycle through batches from dataloader
    for batch_idx, (data, target) in enumerate(train_loader):
        # Transfer to target device
        data, target = data.to(device), target.to(device)
        
        # Check dimensionality of data
        if data.ndim == 2:
            data = data.unsqueeze(0) # If only 1 trial, add dimension
            
        # Reset model hidden state
        model.initHidden(data.shape[0])
        
        # Fill tensors with actual & reconstructed data for comparison
        reconstructed = torch.zeros(data.shape)
        actual = torch.zeros(data.shape)
        
        # Loop through each timepoint
        for t in range(data.shape[1]):
            # Get current timepoint
            data_t = data[:, t, :]
            data_t = data_t.unsqueeze(1) # Ensure dimensions still correct
            
            # Get actual & reconstructed data
            reconstructed[:, t, :] = model(data_t).squeeze()
            actual[:, t, :] = data_t.detach().squeeze()
            
        # Get loss for entire time series
        loss = loss_fn(reconstructed, actual)
        
        # Backpropragate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Output average loss for epoch if at logging interval
        if batch_idx % args['log_interval'] == 0 and args['logging'] == True:
            loss_ = loss.item()
            print(f"Train epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx/len(train_loader):.0f}%)]\tLoss: {loss_:.6f}")
            epoch_loss.append(loss_)
            
            # If trial run, stop after single logging event
            if args['dry_run']:
                break
                
    # Return loss for training epoch
    return epoch_loss

# Define testing routine
def test(model, loss_fn, device, test_loader, logging=True):
    '''
    Runs testing routine
    
    Args:
        model: model to use for testing
        loss_fn: loss function for testing model
        device: device to test on
        test_loader: dataloader for testing data
        logging: whether to print log of test logg
    '''
    # Set model to evaluation mode
    model.eval()
    
    # Initialise variable to track loss
    test_loss = 0
    
    # Deactivate gradients (i.e. no updating model parameters)
    with torch.no_grad():
        # Cycle through batches in test loader
        for batch_idx, (data, target) in enumerate(test_loader):
            # Transfer data to target device
            data, target = data.to(device), target.to(device)
            
            # Check dimensionality of data
            if data.ndim == 2:
                data = data.unsqueeze(0) # If only one trial, add dimension
                
            # Reset model hidden state
            model.initHidden(data.shape[0])
            
            # Fill tensors with reconstructed & actual data for comparison
            reconstructed = torch.zeros(data.shape)
            actual = torch.zeros(data.shape)
            
            # Loop over timepoints
            for t in range(data.shape[1]):
                # Get current timepoint
                data_t = data[:, t, :]
                data_t = data_t.unsqueeze(1) # Fix dimensions
                
                # Get reconstructed and actual data at current point
                reconstructed[:, t, :] = model(data_t).squeeze()
                actual[:, t, :] = data_t.detach().squeeze()
                
            # Get loss for whole timeseries
            loss = loss_fn(reconstructed, actual)
            
            # Accumulate test loss
            test_loss += loss.item()
            
    # Get average test loss
    test_loss /= (batch_idx + 1) # Divide by number of batches
    
    # Print test log
    if logging:
        print(f"\nTest set:\nAverage loss: {test_loss:.4f}\n")
        
    # Return average loss for test set
    return test_loss

# Define validation routine
def validate(model, loss_fn, device, validation_loader):
    '''
    Runs validation routine
    
    Args:
        model: model to run validation with
        loss_fn: loss function for calculating validation loss
        device: device to run validation on (CPU or GPU)
        validation_loader: dataloader for validation data
        
    Returns validation loss & actual and reconstructed data for comparison
    '''
    # Set model to evaluation mode
    model.eval()
    
    # Initialise variable to track validation loss
    validation_loss = 0
    
    # Deactivate gradient accumulation
    with torch.no_grad():
        # Cycle through batches in validation loader
        for batch_idx, (data, target) in enumerate(validation_loader):
            # Transfer data to target device
            data, target = data.to(device), target.to(device)
            
            # Check dimensionality of data
            if data.ndim == 2:
                data = data.unsqueeze(0) # If only one trial, add dimension
                
            # Reset model hidden state
            model.initHidden(data.shape[0])
            
            # Fill tensors with reconstructed & actual data for comparison
            if batch_idx == 0:
                reconstructed_all = torch.zeros((1, data.shape[1], data.shape[2]))
                actual_all = torch.zeros((1, data.shape[1], data.shape[2]))

            reconstructed = torch.zeros(data.shape)
            actual = torch.zeros(data.shape)
            
            # Loop over timepoints
            for t in range(data.shape[1]):
                # Get timepoint
                data_t = data[:, t, :]
                data_t = data_t.unsqueeze(1) # Fix dimensions
                
                # Get reconstructed and actual data at current point
                reconstructed[:, t, :] = model(data_t).squeeze()
                actual[:, t, :] = data_t.detach().squeeze()

            # Get loss for whole timeseries
            loss = loss_fn(reconstructed, actual)
            
            # Add to running loss
            validation_loss += loss.item()
            
            # Save all reconstructed data for plotting
            reconstructed_all = torch.cat((reconstructed_all, reconstructed), dim=0)
            actual_all = torch.cat((actual_all, actual), dim=0)
            
    # Get average validation loss
    validation_loss /= (batch_idx + 1) # Divide by number of batches
    
    # Return average loss & reconstructed data for examples
    reconstructed_all = reconstructed_all[1:, :, :]
    actual_all = actual_all[1:, :, :]
    return validation_loss, (actual_all, reconstructed_all)

# Define routine for running model training
def run_training(training_args, filename='model.pt'):
    '''
    Run training using model parameters specified
    
    Args:
        training_args: dict of training parameters
            args: dict of model parameters
                batch_size: number of samples in each training batch
                test_batch_size: number of samples in each test batch
                epochs: number of epochs to train
                lr: initial learning rate **currently disabled**
                gamma: learning rate step **currently disabled**
                no_cuda: whether to disable CUDA training
                dry_run: whether to stop after single log (for debugging)
                seed: random seed for reproducibility
                logging: whether to output log of loss during training
                log_interval: how often to log loss (in batches)
                save_model: whether to save model to file
            model: model to train
            loss_fn: loss function to use
            device: device to train on
            train_loader: dataloader for training data
            test_loader: dataloader to test data
            optimizer: optimizer to use for model parameters
            epoch: current epoch
        filename: filename to save model to
    '''
    # Create lists for tracking loss
    train_loss = []
    test_loss = []
    
    # Create arguments to pass to train & test
    train_args = training_args.copy()
    train_args.pop('test_loader', None) # Remove test data from training parameters
    test_args = {
        'model': training_args['model'],
        'loss_fn': training_args['loss_fn'],
        'device': training_args['device'],
        'test_loader': training_args['test_loader'],
        'logging': training_args['args']['logging']
    }
    
    # Cycle through epochs
    for epoch in range(1, training_args['args']['epochs']+1):
        # Set current epoch
        train_args.update({'epoch': epoch})
        
        # Run training
        epoch_train_loss = train(**train_args)
        train_loss.append(epoch_train_loss)
        
        # Run testing
        #epoch_test_loss = test(**test_args)
        epoch_test_loss = 0 # Skip running test routine currently
        test_loss.append(epoch_test_loss)
        
    # Save model if required
    if training_args['args']['save_model']:
        torch.save(training_args['model'].state_dict(), filename)
        
    # Return loss
    return (train_loss, test_loss)

# Define routine for fitting model
def fit_model(model, data, params):
    '''
    Fits model to data
    
    Args:
        model: model to fit
        data: data dict (train/test/validate)
        params: dict of parameters for training
        
    Returns model after training
    '''
    # Set up training parameters
    batch_sizes = params['batch_sizes']
    
    # Set up model parameters
    model_args = {
        'batch_size': batch_sizes['train'],
        'epochs': params['epochs'],
        'lr': 1.0,
        'gamma': 0.7,
        'no_cuda': False,
        'dry_run': False,
        'seed': 17,
        'logging': True,
        'log_interval': 200,
        'save_model': False
    }
    
    # Determine whether or not to use GPU
    use_cuda = torch.cuda.is_available()
    device = params['device']
    
    # Set dataloader arguments
    dataloader_args = {'batch_size': model_args['batch_size']}
    if use_cuda:
        dataloader_args.update({
            'num_workers': 1,
            'pin_memory': True
        })
    
    # Create dataloaders
    train_data = HandData(data['X_train'], data['Y_train'])
    train_loader = DataLoader(train_data, **dataloader_args)
    
    dataloader_args.update({'batch_size': batch_sizes['test']})
    test_data = HandData(data['X_test'], data['Y_test'])
    test_loader = DataLoader(test_data, **dataloader_args)
    
    # Transfer model to target device
    model = model.to(device)
    
    # Instantiate optimizer
    optimizer = optim.Adam(model.parameters())
    
    # Instantiate loss function
    loss_fn = nn.MSELoss()
    
    # Set training parameters
    training_args = {
        'args': model_args,
        'model': model,
        'loss_fn': loss_fn,
        'device': device,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'epoch': 0
    }
    
    # Run training
    train_loss, test_loss = run_training(training_args)
    
    # Return model
    return model

# Define model validation routine
def validate_model(model, data, params):
    '''
    Runs validation for fit model
    
    Args:
        model: model to run validation for
        data: data dict (train/test/validate)
        params: dict of parameters for training
    '''
    # Determine whether or not to use GPU
    use_cuda = torch.cuda.is_available()
    device = params['device']
    
    # Set dataloader arguments
    dataloader_args = {'batch_size': params['batch_sizes']['validate']}
    if use_cuda:
        dataloader_args.update({
            'num_workers': 1,
            'pin_memory': True
        })
        
    # Prepare dataloader
    validation_data = HandData(data['X_validate'], data['Y_validate'])
    validation_loader = DataLoader(validation_data, **dataloader_args)
   
    # Set loss function
    loss_fn = nn.MSELoss()
    
    # Run validation
    validation_loss, (actual, reconstructed) = validate(model, loss_fn, device, validation_loader)
    
    # Return loss
    return validation_loss, (actual, reconstructed)
