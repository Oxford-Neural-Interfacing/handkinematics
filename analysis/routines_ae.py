'''
Routines for analysis hand biomechanics with autoencoder model

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
27/08/2021

Runs analysis for assessing performance of autoencoder models
'''
# Imports
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV as RandomisedSearchCV
from scipy.stats import uniform

from network import *

# Reconstruction analysis
def reconstruct_ae(data:dict, args:dict) -> dict:
    '''
    Evaluates ability of autoencoder model to reconstruct hand biomechanics data
    
    Args:
        data (dict): data to use (train/test/validate splits)
        args (dict): other parameters required to run analysis
        
    Returns:
        results (dict): results of analyses
    '''
    # Prepare model
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    params = {
        'device': device,
        'batch_sizes': {
            'train': 32*15,
            'test': 32*15,
            'validate': 32*15
        },
        'epochs': 250 # 32, 100: 10mins roughly (250: 20 mins)
    }
    
    batch_size = params['batch_sizes']['train'] # Hyperparameter; ?add as argument to pass between functions
    input_size = data['X_train'].shape[2] # Number of channels of kinematic data
    hidden_size = 64 # Size of LSTM hidden layer; hyperparameter
    num_dims = args['num_dims']
    
    model = AutoEncoder(batch_size, input_size, hidden_size, num_dims, device).to(device)
    
    # Train model
    model = fit_model(model, data, params)
    args.update({'model_ae': model}) # Save model for use in classification
    
    # Get validation loss
    mse_ae, (actual, reconstructed) = validate_model(model, data, params)
    
    # Save examples of reconstruction for plotting
    num_examples = 4
    trials_example = np.random.randint(data['X_validate'].shape[0], size=num_examples)
    sensors_example = np.random.randint(input_size, size=num_examples)
    example_actual_ae = np.zeros((num_examples, data['X_validate'].shape[1]))
    example_reconstructed_ae = np.zeros((num_examples, data['X_validate'].shape[1]))
    for i in range(num_examples):
        example_actual_ae[i, :] = actual[trials_example[i], :, sensors_example[i]]
        example_reconstructed_ae[i, :] = reconstructed[trials_example[i], :, sensors_example[i]]
        
    # Get loss over range of dimensions
    X_train = data['X_train']
    Y_train = data['Y_train']
    Y_test = data['Y_validate']

    X_train_classify = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))

    # Create classifier to use for assess accuracy vs. dimensions
    classifier = RandomisedSearchCV(
        LogisticRegression(max_iter=250),
        {'C': uniform(loc=0, scale=4)},
        n_iter=25,
        scoring='accuracy',
        cv=5,
        refit=True,
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_train_classify, Y_train)

    #params.update({'epochs': 100})
    mse_ae_dims = np.zeros(11) # Keep only every second - ?compute from input_size
    accuracy_ae_dims = np.zeros(11)
    explained_variance_ae_dims = np.zeros(11)
    i = 0
    for dims in range(input_size):
        # Run only for every second dimension (1, 3, 5, .., 21)
        if dims % 2 == 0: # 0th dim -> 1 dimension
            model = AutoEncoder(batch_size, input_size, hidden_size, dims+1, device).to(device)
            model = fit_model(model, data, params)
            mse_ae_dims[i], (actual, reconstructed_dims) = validate_model(model, data, params)

            dim1 = reconstructed_dims.shape[0]
            dim2 = reconstructed_dims.shape[1]
            dim3 = reconstructed_dims.shape[2]
            
            reconstructed_dims = torch.reshape(reconstructed_dims, (reconstructed_dims.shape[0], reconstructed_dims.shape[1]*reconstructed_dims.shape[2]))
            reconstructed_dims = reconstructed_dims.detach().numpy()
            
            predicted = classifier.predict(reconstructed_dims)
            accuracy_ae_dims[i] = accuracy_score(Y_test, predicted)
            
            actual = torch.reshape(actual, (actual.shape[0]*actual.shape[1], actual.shape[2]))
            actual = actual.detach().numpy()
            reconstructed_dims = np.reshape(reconstructed_dims, (dim1*dim2, dim3))
            
            explained_variance_ae_dims[i] = explained_variance_score(actual, reconstructed_dims)

            # Increment index
            i = i + 1
        
    # Define results to return
    results = {
        'mse_ae': mse_ae,
        'mse_ae_dims': mse_ae_dims,
        'accuracy_ae_dims': accuracy_ae_dims,
        
        'example_actual_ae': example_actual_ae,
        'example_reconstructed_ae': example_reconstructed_ae,
        
        'explained_variance_ae_dims': explained_variance_ae_dims,
        
        # Also keep full set of reconstructed data for assessment & plotting (more flexible than just examples)
        'actual_ae': actual, # Should be same as actual_linear (as same data)
        'reconstructed_ae': reconstructed
    }
    return results
    
# Classification analysis
def classify_ae(data:dict, args:dict) -> dict:
    '''
    Evaluates ability of autoencoder model to classify hand biomechanics data
    
    Args:
        data (dict): data to use (train/test/validate splits)
        args (dict): other parameters required to run analysis
        
    Returns:
        results (dict): results of analyses
    '''
    # Get data required
    X_train = torch.Tensor(data['X_train'])
    X_test = torch.Tensor(data['X_validate'])
    Y_train = torch.Tensor(data['Y_train'])
    Y_test = torch.Tensor(data['Y_validate'])
    
    # Get parameters
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    num_t = X_train.shape[1]
    num_sensors = X_train.shape[2]
    
    # Prepare model
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    params = {
        'device': device,
        'batch_sizes': {
            'train': 32*15,
            'test': 32*15,
            'validate': 32*15
        },
        'epochs': 250
    }
    
    batch_size = params['batch_sizes']['train'] # Hyperparameter; ?add as argument to pass between functions
    hidden_size = 64 # Size of LSTM hidden layer; hyperparameter
    num_dims = args['num_dims']
    
    model = AutoEncoder(batch_size, num_sensors, hidden_size, num_dims, device).to(device)
    
    # Train model
    #model = fit_model(model, data, params)
    model = args['model_ae'] # Use previously trained model (from reconstruction)
    
    # Encode data
    #X_train_reduced = torch.zeros((train_size, num_t, num_dims))
    #X_test_reduced = torch.zeros((test_size, num_t, num_dims))
    X_test_reconstructed = torch.zeros((test_size, num_t, num_sensors)) # Use reconstructed instead of encoded

    X_train = X_train.to(device)
    X_test = X_test.to(device)

    #model.initHidden(train_size)
    #for t in range(num_t):
        #x_ = X_train[:, t, :]
        #x_ = x_.unsqueeze(1)
        #X_train_reduced[:, t, :] = model.encode(x_).squeeze()
        
    model.initHidden(test_size)
    for t in range(num_t):
        x_ = X_test[:, t, :]
        x_ = x_.unsqueeze(1)
        #X_test_reduced[:, t, :] = model.encode(x_).squeeze()
        X_test_reconstructed[:, t, :] = model(x_).squeeze()

    # Reshape: trials x [time x num_dims]
    #X_train_reduced = torch.reshape(X_train_reduced, (train_size, num_t*num_dims))
    #X_test_reduced = torch.reshape(X_test_reduced, (test_size, num_t*num_dims))
    X_train_classify = torch.reshape(X_train, (train_size, num_t*num_sensors))
    X_test_classify = torch.reshape(X_test, (test_size, num_t*num_sensors))
    X_test_reconstructed = torch.reshape(X_test_reconstructed, (test_size, num_t*num_sensors))

    # Convert to numpy array
    #X_train_reduced = X_train_reduced.detach().numpy()
    #X_test_reduced = X_test_reduced.detach().numpy()
    X_train_classify = X_train_classify.cpu().detach().numpy()
    X_test_classify = X_test_classify.cpu().detach().numpy()
    X_test_reconstructed = X_test_reconstructed.cpu().detach().numpy()
    Y_train = Y_train.detach().numpy()
    Y_test = Y_test.detach().numpy()
    
    # Define classifiers - model, parameter distributions, title
    log_reg = {
        'Title': 'Softmax regression',
        'model': LogisticRegression(max_iter=250), # 1000 - as for other
        'params': {
            'C': uniform(loc=0, scale=4)
        }
    }
    
    svm = {
        'Title': 'Support vector machine',
        'model': SVC(max_iter=250),
        'params': {
            'C': uniform(loc=0, scale=4)
        }
    }
    
    sgd = {
        'Title': 'SGD classifier',
        'model': SGDClassifier(random_state=42, max_iter=250),
        'params': {
            'loss': ['hinge', 'log'], # Hinge = SVM, log = LR
            'alpha': uniform(loc=0, scale=5)
        }
    }
    
    nb = {
        'Title': 'Naive Bayes',
        'model': GaussianNB(),
        'params': {}
    }
    
    forest = {
        'Title': 'Random forest',
        'model': RandomForestClassifier(random_state=42),
        'params': {}
    }
    
    # Combine models into list
    classifiers = [
        log_reg,
        svm,
        sgd,
        nb,
        forest
    ]
    
    # Iterate over list & run randomised parameter search (leabe out last two as no parameters to optimise)
    for classifier in classifiers[:3]:
        classifier.update({
            'best': RandomisedSearchCV(
            classifier['model'], classifier['params'],
            n_iter=25, # 50 (orig 100)
            scoring='accuracy',
            cv=5,
            refit=True,
            random_state=42,
            n_jobs=-1
            )
        })
        #classifier['best'].fit(X_train_reduced, Y_train)
        classifier['best'].fit(X_train_classify, Y_train)
        
    # Add other models
    for classifier in classifiers[3:]:
        classifier.update({
            'best': classifier['model'] # Just use default as no parameters to optimise
        })
        #classifier['best'].fit(X_train_reduced, Y_train)
        classifier['best'].fit(X_train_classify, Y_train)
        
    # Get accuracy of each classifier on test data
    accuracy = np.zeros(len(classifiers))
    accuracy_baseline = np.zeros(len(classifiers))
    for idx, classifier in enumerate(classifiers):
        #predicted = classifier['best'].predict(X_test_reduced)
        predicted = classifier['best'].predict(X_test_reconstructed)
        accuracy[idx] = accuracy_score(Y_test, predicted)

        predicted_baseline = classifier['best'].predict(X_test_classify)
        accuracy_baseline[idx] = accuracy_score(Y_test, predicted_baseline)
        
    # Add classifier to args - ?determine best vs. just use LR
    args.update({'classifier': log_reg})
    
    # Define results to return
    results = {
        'accuracy_ae': accuracy,
        'accuracy_baseline_ae': accuracy_baseline,
        'accuracy_ae_max': np.max(accuracy),
        'accuracy_baseline_ae_max': np.max(accuracy_baseline)
    }
    return results

# Interpolation analysis
def interpolate_ae(data:dict, args:dict) -> dict:
    '''
    Evaluates ability of linear model to interpolate unseen movements
    
    Args:
        data (dict): data to use (train/test/validate splits)
        args (dict): other parameters required to run analysis
        
    Returns:
        results (dict): results of analyses
    '''
    # Get data required
    # If transfer analysis, separate this participant's data from others; otherwise combine all
    if args['is_transfer']:
        X_other_participants = np.concatenate((data['X_train'], data['X_test']), axis=0)
        Y_other_participants = np.concatenate((data['Y_train'], data['Y_test']), axis=0)
        
        X_this_participant = data['X_validate']
        Y_this_participant = data['Y_validate']
    else:
        X = np.concatenate((data['X_train'], data['X_test'], data['X_validate']), axis=0)
        Y = np.concatenate((data['Y_train'], data['Y_test'], data['Y_validate']), axis=0)
    
    # List all tasks
    tasks = args['task_list']
    interpolation_loss = np.zeros(len(tasks))
    
    # Iterate over tasks & calculate interpolation loss
    for idx, task in enumerate(tasks):
        # Remove all trials with specific task
        if args['is_transfer']:
            X_task = X_this_participant[Y_this_participant==task, :, :]
            X_others = X_other_participants[Y_other_participants!=task, :, :]
            Y_task = Y_this_participant[Y_this_participant==task]
            Y_others = Y_other_participants[Y_other_participants!=task]
            
            # Save all reconstructed data for comparison with standard reconstructions
            reconstructed_interpolated = np.zeros_like(X_this_participant)
        else:
            X_task = X[Y==task, :, :]
            X_others = X[Y!=task, :, :]
            Y_task = Y[Y==task]
            Y_others = Y[Y!=task]
            
            # Save all reconstructed data for comparison with standard reconstructions
            reconstructed_interpolated = np.zeros_like(X)
        
        task_size = X_task.shape[0]
        others_size = X_others.shape[0]
        num_t = X_task.shape[1]
        num_sensors = X_task.shape[2]
        
        
        # Reshape data
        X_task = np.reshape(X_task, (task_size*num_t, num_sensors))
        X_others = np.reshape(X_others, (others_size*num_t, num_sensors))
        
        # Rescale data
        scaler = MinMaxScaler()
        scaler.fit(X_others)
        X_others = scaler.transform(X_others)
        X_task = scaler.transform(X_task)
        
        # Reshape back to trials x time x sensors
        X_task = np.reshape(X_task, (task_size, num_t, num_sensors))
        X_others = np.reshape(X_others, (others_size, num_t, num_sensors))
        
        # Fit model to other tasks
        task_data = {
            'X_train': X_others,
            'X_test': X_task, # Not used for anything; need for consistency
            'X_validate': X_task,
            'Y_train': Y_others, # Not used for anything; need for consistency
            'Y_test': Y_task,
            'Y_validate': Y_task
        }

        # Prepare model
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')

        params = {
            'device': device,
            'batch_sizes': {
                'train': 32*15,
                'test': 32*15,
                'validate': 32*15
            },
            'epochs': 250 
        }

        batch_size = params['batch_sizes']['train'] # Hyperparameter; ?add as argument to pass between functions
        hidden_size = 64 # Size of LSTM hidden layer; hyperparameter
        num_dims = args['num_dims']

        model = AutoEncoder(batch_size, num_sensors, hidden_size, num_dims, device).to(device)

        # Train model
        model = fit_model(model, task_data, params)
        
        # Get interpolation loss
        interpolation_loss[idx], (actual, reconstructed) = validate_model(model, task_data, params)
       
        # Save reconstructions
        if args['is_transfer']:
            reconstructed_interpolated[Y_this_participant==task, :, :] = reconstructed
        else:
            reconstructed_interpolated[Y==task, :, :] = reconstructed
        
    # Rank tasks by interpolation loss
    ranked_tasks = [x for _, x in sorted(zip(interpolation_loss, tasks), key=lambda pair: pair[0], reverse=True)]
    ranked_loss = sorted(interpolation_loss, reverse=True)
    num_tasks = np.arange(1, len(ranked_tasks)+1)
    interp_loss_tasknum = np.zeros(int(len(num_tasks)//2)) # Keep only every second
    interp_accuracy_tasknum = np.zeros(int(len(num_tasks)//2))
   
    # Get train & test data
    X_train = np.concatenate((data['X_train'], data['X_test']), axis=0)
    X_test = data['X_validate']
    Y_train = np.concatenate((data['Y_train'], data['Y_test']), axis=0)
    Y_test = data['Y_validate']
    
    # Iterate over number of tasks to include
    i = 0 # Using only every second -> increment manually
    for i_, n in enumerate(num_tasks):
        # Include only every second set, i.e. (2, 4, ... 14) tasks
        if n % 2 == 0:
            # Get included tasks
            included_tasks = ranked_tasks[:n]
        
            # Get trials to include
            included_inds = [i for i, y in enumerate(Y_train) if y in included_tasks]
            interp_X_train = X_train[included_inds, :, :]
            interp_X_test = X_test
        
            train_size = interp_X_train.shape[0]
            test_size = interp_X_test.shape[0]
        
            # Reshape data
            interp_X_train = np.reshape(interp_X_train, (train_size*num_t, num_sensors))
            interp_X_test = np.reshape(interp_X_test, (test_size*num_t, num_sensors))
        
            # Rescale data
            scaler = MinMaxScaler()
            scaler.fit(interp_X_train)
            interp_X_train = scaler.transform(interp_X_train)
            interp_X_test = scaler.transform(interp_X_test)
        
            # Reshape back to trials x time x sensors
            interp_X_train = np.reshape(interp_X_train, (train_size, num_t, num_sensors))
            interp_X_test = np.reshape(interp_X_test, (test_size, num_t, num_sensors))
        
            # Fit model to other tasks
            task_data = {
                'X_train': interp_X_train,
                'X_test': interp_X_test, # Not used for anything; need for consistency
                'X_validate': interp_X_test,
                'Y_train': Y_train,
                'Y_test': Y_test,
                'Y_validate': Y_test
            }

            # Prepare model
            use_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')

            params = {
                'device': device,
                'batch_sizes': {
                    'train': 32*15,
                    'test': 32*15,
                    'validate': 32*15
                },
                'epochs': 250
            }

            batch_size = params['batch_sizes']['train'] # Hyperparameter; ?add as argument to pass between functions
            hidden_size = 64 # Size of LSTM hidden layer; hyperparameter
            num_dims = args['num_dims']

            model = AutoEncoder(batch_size, num_sensors, hidden_size, num_dims, device).to(device)

            # Train model
            model = fit_model(model, task_data, params)
        
            # Get interpolation loss
            interp_loss_tasknum[i], _ = validate_model(model, task_data, params)
        
            # Get classification data on test data

            # Use reconstructed rather than encoded - use model trained on full test data, i.e. no re-train necessary
            '''
            train_size = interp_X_train.shape[0]
            interp_X_train_reconstructed = torch.zeros((train_size, num_t, num_sensors)).to(device)
            interp_X_train = torch.Tensor(interp_X_train).to(device)
            model.initHidden(interp_X_train.shape[0])
            for t in range(num_t):
                x_ = interp_X_train[:, t, :]
                x_ = x_.unsqueeze(1)
                interp_X_train_reconstructed[:, t, :] = model(x_).squeeze()

            interp_X_train_reduced = torch.reshape(interp_X_train_reduced, (train_size, num_t*num_dims))
            interp_X_train_reduced = interp_X_train_reduced.cpu().detach().numpy()
            '''

            test_size = interp_X_test.shape[0]
            interp_X_test_reconstructed = torch.zeros((test_size, num_t, num_sensors)).to(device)
            interp_X_test = torch.Tensor(interp_X_test).to(device)
            model.initHidden(interp_X_test.shape[0])
            for t in range(num_t):
                x_ = interp_X_test[:, t, :]
                x_ = x_.unsqueeze(1)
                interp_X_test_reconstructed[:, t, :] = model(x_).squeeze()
        
            interp_X_test_reconstructed = torch.reshape(interp_X_test_reconstructed, (test_size, num_t*num_sensors))
            interp_X_test_reconstructed = interp_X_test_reconstructed.cpu().detach().numpy()

            predicted = args['classifier']['best'].predict(interp_X_test_reconstructed)
            interp_accuracy_tasknum[i] = accuracy_score(Y_test, predicted)

            # Increment index
            i = i+1
        
    # Define results to return
    results = {
        'ae_interpolation_loss': interpolation_loss,
        'ae_interpolation_loss_mean': np.mean(interpolation_loss),
        'ae_ranked_tasks': ranked_tasks,
        'ae_ranked_loss': ranked_loss,
        'ae_interpolation_loss_tasknum': interp_loss_tasknum,
        'ae_interpolation_accuracy_tasknum': interp_accuracy_tasknum,
        'ae_reconstructed_interpolated': reconstructed_interpolated
    }
    return results
