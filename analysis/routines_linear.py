'''
Routines for analysing hand biomechanics data with linear model

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
09/08/2021

Runs analysis for assessing performance of linear models
'''
# Imports
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
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

# Reconstruction analysis
def reconstruct_linear(data:dict, args:dict) -> dict:
    '''
    Evaluates ability of linear model to reconstruct hand biomechanics data
    
    Args:
        data (dict): data to use (train/test/validate splits)
        args (dict): other parameters required to run analysis
        
    Returns:
        results (dict): results of analyses
    '''
    # Get data required
    train = data['X_train']
    test = data['X_validate']
    
    # Define PCA model
    pca = PCA(n_components=args['num_dims']) # Number of dimensions determined by args dict
    
    # Prepare data
    train_size = train.shape[0]
    test_size = test.shape[0]
    num_t = train.shape[1]
    num_sensors = train.shape[2]
    train_reshaped = np.reshape(train, (train_size*num_t, num_sensors))
    test_reshaped = np.reshape(test, (test_size*num_t, num_sensors))
    
    # Fit model
    pca.fit(train_reshaped)
    
    # Reduce test data
    test_reduced = pca.transform(test_reshaped)
    
    # Reconstruct test data
    reconstructed = pca.inverse_transform(test_reduced)
    
    # Get MSE
    mse_linear = mean_squared_error(test_reshaped, reconstructed)
    
    # Reshape data to trials x time x sensors
    reconstructed = np.reshape(reconstructed, (test_size, num_t, num_sensors))
    reconstructed_ = reconstructed # Keep for saving & plotting
    
    # Save example of reconstruction for plotting
    num_examples = 4 # Number of trials to save
    trials_example = np.random.randint(test_size, size=num_examples)
    sensors_example = np.random.randint(num_sensors, size=num_examples)
    example_actual = np.zeros((num_examples, num_t))
    example_reconstructed_linear = np.zeros((num_examples, num_t))
    for i in range(num_examples):
        example_actual[i, :] = test[trials_example[i], :, sensors_example[i]]
        example_reconstructed_linear[i, :] = reconstructed[trials_example[i], :, sensors_example[i]]
    
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

    mse_linear_dims = np.zeros(11) # Keep only every second - ?compute from num_sensors
    accuracy_linear_dims = np.zeros(11)
    explained_variance_linear_dims = np.zeros(11)
    i = 0 # Index manually
    for dims in range(num_sensors):
        # Run only for every second dimension (1, 3, 5, ... 21)
        if dims % 2 == 0:
            pca = PCA(n_components=dims+1)
            pca.fit(train_reshaped)
            test_reduced = pca.transform(test_reshaped)
            reconstructed = pca.inverse_transform(test_reduced)
            mse_linear_dims[i] = mean_squared_error(test_reshaped, reconstructed)

            reconstructed = np.reshape(reconstructed, (test_size, num_t, num_sensors))
            reconstructed = np.reshape(reconstructed, (test_size, num_t*num_sensors))

            predicted = classifier.predict(reconstructed)
            accuracy_linear_dims[i] = accuracy_score(Y_test, predicted)
            
            reconstructed = np.reshape(reconstructed, (test_size*num_t, num_sensors))
            explained_variance_linear_dims[i] = explained_variance_score(test_reshaped, reconstructed)

            # Increment index
            i = i + 1
        
    # Define results to return
    results = {
        'mse_linear': mse_linear,
        'mse_linear_dims': mse_linear_dims,
        'accuracy_linear_dims': accuracy_linear_dims,
        
        'example_actual': example_actual,
        'example_reconstructed_linear': example_reconstructed_linear,
        
        'explained_variance_linear_dims': explained_variance_linear_dims,
        
        # Also keep full set of reconstructed data for assessment & plotting (more flexible than just examples)
        'actual_linear': test,
        'reconstructed_linear': reconstructed_
    }
    return results

# Classification analysis
def classify_linear(data:dict, args:dict) -> dict:
    '''
    Evaluates ability of linear model to classify hand biomechanics data
    
    Args:
        data (dict): data to use (train/test/validate splits)
        args (dict): other parameters required to run analysis
        
    Returns:
        results (dict): results of analyses
    '''
    # Get data required
    X_train = data['X_train']
    X_test = data['X_validate']
    Y_train = data['Y_train']
    Y_test = data['Y_validate']
    
    # Prepare data: trials x [time x num_dims]
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    num_t = X_train.shape[1]
    num_sensors = X_train.shape[2]
    X_train_reshaped = np.reshape(X_train, (train_size*num_t, num_sensors))
    X_test_reshaped = np.reshape(X_test, (test_size*num_t, num_sensors))

    # Run linear model on all data
    pca = PCA(n_components=args['num_dims'])
    pca.fit(X_train_reshaped)

    # Use reconstructed rather than encoded data
    #X_train_reduced = pca.transform(X_train_reshaped)
    #X_train_reduced = np.reshape(X_train_reduced, (train_size, num_t, args['num_dims']))
    X_test_reduced = pca.transform(X_test_reshaped)
    #X_test_reduced = np.reshape(X_test_reduced, (test_size, num_t, args['num_dims']))
    X_test_reconstructed = pca.inverse_transform(X_test_reduced)
    X_test_reconstructed = np.reshape(X_test_reconstructed, (test_size, num_t, num_sensors))
    
    # Prepare final shape: trials x [time x num_dims]
    #X_train_reduced = np.reshape(X_train_reduced, (train_size, num_t*args['num_dims']))
    #X_test_reduced = np.reshape(X_test_reduced, (test_size, num_t*args['num_dims']))
    X_train_classify = np.reshape(X_train, (train_size, num_t*num_sensors))
    X_test_classify = np.reshape(X_test, (test_size, num_t*num_sensors))
    X_test_reconstructed = np.reshape(X_test_reconstructed, (test_size, num_t*num_sensors))
    
    # Define classifiers - model, parameter distributions, title
    log_reg = {
        'Title': 'Softmax regression',
        'model': LogisticRegression(max_iter=250), # 1000 - as other
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
            n_iter=25,
            scoring='accuracy',
            cv=5,
            refit=True,
            random_state=42,
            n_jobs=-1
            )
        })
        classifier['best'].fit(X_train_classify, Y_train)
        
    # Add other models
    for classifier in classifiers[3:]:
        classifier.update({
            'best': classifier['model'] # Just use default as no parameters to optimise
        })
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
        'accuracy_linear': accuracy,
        'accuracy_baseline_linear': accuracy_baseline,
        'accuracy_linear_max': np.max(accuracy),
        'accuracy_baseline_linear_max': np.max(accuracy_baseline)
    }
    return results

# Interpolation analysis
def interpolate_linear(data:dict, args:dict) -> dict:
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
            
            # Save all reconstructed data for comparison with standard reconstructions
            reconstructed_interpolated = np.zeros_like(X_this_participant)
        else:
            X_task = X[Y==task, :, :]
            X_others = X[Y!=task, :, :]
            
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
        
        # Run PCA
        pca = PCA(n_components=args['num_dims'])
        pca.fit(X_others)
        X_task_reduced = pca.transform(X_task)
        X_task_reconstructed = pca.inverse_transform(X_task_reduced)
        
        # Get interpolation loss
        interpolation_loss[idx] = mean_squared_error(X_task, X_task_reconstructed)
        
        # Save reconstructions
        X_task_reconstructed = np.reshape(X_task_reconstructed, (task_size, num_t, num_sensors))
        if args['is_transfer']:
            reconstructed_interpolated[Y_this_participant==task, :, :] = X_task_reconstructed
        else:
            reconstructed_interpolated[Y==task, :, :] = X_task_reconstructed
        
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
    i = 0 # Manually index - keeping only every second item
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
        
            # PCA data
            pca = PCA(n_components=args['num_dims'])
            pca.fit(interp_X_train)

            interp_X_test_reduced = pca.transform(interp_X_test)
            interp_X_test_reconstructed = pca.inverse_transform(interp_X_test_reduced)
        
            # Get loss
            interp_loss_tasknum[i] = mean_squared_error(interp_X_test, interp_X_test_reconstructed)
        
            # Get classification accuracy on test data
            interp_X_test_reconstructed = np.reshape(interp_X_test_reconstructed, (test_size, num_t, num_sensors))
            interp_X_test_reconstructed = np.reshape(interp_X_test_reconstructed, (test_size, num_t*num_sensors))

            predicted = args['classifier']['best'].predict(interp_X_test_reconstructed)
            interp_accuracy_tasknum[i] = accuracy_score(Y_test, predicted)

            # Increment index
            i = i+1
        
    # Define results to return
    results = {
        'linear_interpolation_loss': interpolation_loss,
        'linear_interpolation_loss_mean': np.mean(interpolation_loss),
        'linear_ranked_tasks': ranked_tasks,
        'linear_ranked_loss': ranked_loss,
        'linear_interpolation_loss_tasknum': interp_loss_tasknum,
        'linear_interpolation_accuracy_tasknum': interp_accuracy_tasknum,
        'linear_reconstructed_interpolated': reconstructed_interpolated
    }
    return results
