'''
Routines for plotting hand biomechanics results

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
13/08/2021

Creates & saves plots for individual & group results
'''
# Import
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('talk')

from matplotlib.lines import Line2D

# Set palette
palette = [
    '#002147', # Oxford Blue
    '#C65400', # Orange
    '#28502E', # Hunter Green
    '#E54B4B', # Imperial Red
    '#A3C1AD'  # Cambridge Blue
]
sns.set_palette(palette)

# Saving - utility function
def saveFig(fig, fname):
    '''
    Saves figure to specified directory
    Closes figure
    Convenience function for generate_figures
    '''
    # Set target directory
    target_dir = '../results/figures'
    
    # Save figure
    fig.savefig(f"{target_dir}/{fname}.png")
    
    # Close figure
    fig.clf()

# Individual plots
def plotting_individual(results:dict, filename:str):
    '''
    Creates & saves plots for individuals
    
    Args:
        results (dict): results dict with individual results
        filename (str): base filename for saving plots
        
    Creates & saves all individual-level plots from results dict
    '''
    ### Example reconstructions ###
    # Get examples - actual, linear, DL
    actual = results['actual_linear']
    reconstructed_linear = results['reconstructed_linear']
    reconstructed_ae = results['reconstructed_ae']
    
    num_examples = 4
    trials_example = np.random.randint(actual.shape[0], size=num_examples)
    sensors_example = np.random.randint(actual.shape[2], size=num_examples)
    
    example_actual = np.zeros((num_examples, actual.shape[1]))
    example_reconstructed_linear = np.zeros((num_examples, actual.shape[1]))
    example_reconstructed_ae = np.zeros((num_examples, actual.shape[1]))
    
    for i in range(num_examples):
        example_actual[i, :] = actual[trials_example[i], :, sensors_example[i]]
        example_reconstructed_linear[i, :] = reconstructed_linear[trials_example[i], :, sensors_example[i]]
        example_reconstructed_ae[i, :] = reconstructed_ae[trials_example[i], :, sensors_example[i]]
    
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(20, 10))
    
    t = np.linspace(0, 2, num=256)
    ax[0,0].plot(t, example_actual[0,:])
    ax[0,0].plot(t, example_reconstructed_linear[0,:])
    ax[0,0].plot(t, example_reconstructed_ae[0,:])
    
    ax[0,1].plot(t, example_actual[1,:])
    ax[0,1].plot(t, example_reconstructed_linear[1,:])
    ax[0,1].plot(t, example_reconstructed_ae[1,:])
    
    ax[1,0].plot(t, example_actual[2,:])
    ax[1,0].plot(t, example_reconstructed_linear[2,:])
    ax[1,0].plot(t, example_reconstructed_ae[2,:])
    
    ax[1,1].plot(t, example_actual[3,:])
    ax[1,1].plot(t, example_reconstructed_linear[3,:])
    ax[1,1].plot(t, example_reconstructed_ae[3,:])
    
    ax[0,0].set(title='Example 1')
    ax[0,1].set(title='Example 2')
    ax[1,0].set(title='Example 3', xlabel='Time (s)')
    ax[1,1].set(title='Example 4', xlabel='Time (s)')
    
    fig.legend(['Actual', 'PCA', 'RNN-AE'], loc='upper right')
    fig.tight_layout(rect=[0,0.03,1,0.95])
    
    fig.savefig(f"../results/{filename}_examples.png")
    fig.clf()
    
    ### MSE vs. dims ###
    fig, ax = plt.subplots()
    x_ = (np.arange(1, len(results['mse_linear_dims'])+1)*2)-1 # Every 2nd value
    ax.plot(x_, results['mse_linear_dims'])
    ax.plot(x_, results['mse_ae_dims'])
    
    ax.legend(['PCA', 'RNN-AE'], loc='upper right')
    ax.set(title='Reconstruction loss', xlabel='Dimensions', ylabel='MSE')
    fig.tight_layout()
    fig.savefig(f"../results/{filename}_mse_dims.png")
    fig.clf()
    
    ### Accuracy vs. dims ###
    fig, ax = plt.subplots()
    x_ = (np.arange(1, len(results['accuracy_linear_dims'])+1)*2)-1 # Every 2nd value
    ax.plot(x_, results['accuracy_linear_dims'])
    ax.plot(x_, results['accuracy_ae_dims'])
    
    ax.legend(['PCA', 'RNN-AE'], loc='upper right')
    ax.set(title='Reconstruction loss', xlabel='Dimensions', ylabel='Accuracy')
    fig.tight_layout()
    fig.savefig(f"../results/{filename}_accuracy_dims.png")
    fig.clf()
    
    ### Classifier performance ###
    fig, ax = plt.subplots()
    
    classifiers = ['LR', 'SVM', 'SGD', 'NB', 'RF']
    x_ = np.array([1, 2, 3, 4, 5])
    w = 0.4
    
    ax.bar(x_-w/2, results['accuracy_linear'], w, label='PCA')
    ax.bar(x_+w/2, results['accuracy_ae'], w, label='RNN-AE')
    
    ax.set(xlabel='Classifier', ylabel='Accuracy')
    ax.legend(loc='upper right')
    plt.xticks([1, 2, 3, 4, 5], classifiers)
    
    fig.tight_layout()
    fig.savefig(f"../results/{filename}_classifiers.png")
    fig.clf()
    
    ### Loss & accuracy vs. tasks ###
    fig, ax = plt.subplots(2, 1, sharex='col')
    num_tasks = np.arange(1, len(results['linear_interpolation_loss_tasknum'])+1)*2 # Every 2nd value
    
    ax[0].plot(num_tasks, results['linear_interpolation_loss_tasknum'])
    ax[0].plot(num_tasks, results['ae_interpolation_loss_tasknum'])
    ax[0].set(title='Reconstruction loss', ylabel='MSE')
    
    ax[1].plot(num_tasks, results['linear_interpolation_accuracy_tasknum'])
    ax[1].plot(num_tasks, results['ae_interpolation_accuracy_tasknum'])
    ax[1].set(title='Classification accuracy', ylabel='Accuracy', xlabel='Number of training tasks')
    
    fig.tight_layout()
    fig.savefig(f"../results/{filename}_lossacc_vstasks.png")
    fig.clf()
    
# Group plots
def plotting_group(results:dict, filename:str):
    '''
    Creates & saves plots for group analyses
    
    Args:
        results (dict): results dict with group results
        filename (str): base filename for saving plots
        
    Creates & saved all group-level plots from results dict
    '''
    ### MSE vs. dims ###
    fig, ax = plt.subplots()
    x_ = (np.arange(1, len(results['mse_linear_dims_mean'])+1)*2)-1
    
    ax.plot(x_, results['mse_linear_dims_mean'])
    ax.fill_between(x_, results['mse_linear_dims_mean']+results['mse_linear_dims_std'], results['mse_linear_dims_mean']-results['mse_linear_dims_std'], alpha=0.2)
    
    ax.plot(x_, results['mse_ae_dims_mean'])
    ax.fill_between(x_, results['mse_ae_dims_mean']+results['mse_ae_dims_std'], results['mse_ae_dims_mean']-results['mse_ae_dims_std'], alpha=0.2)

    ax.legend(['PCA', 'RNN-AE'], loc='upper right')
    ax.set(title='Reconstruction loss', ylabel='MSE', xlabel='Dimensions')
    
    fig.tight_layout()
    fig.savefig(f"../results/{filename}_mse_dims_group.png")
    fig.clf()
    
    ### Accuracy vs. dims ###
    fig, ax = plt.subplots()
    
    ax.plot(x_, results['accuracy_linear_dims_mean'])
    ax.fill_between(x_, results['accuracy_linear_dims_mean']+results['accuracy_linear_dims_std'], results['accuracy_linear_dims_mean']-results['accuracy_linear_dims_std'], alpha=0.2)
    
    ax.plot(x_, results['accuracy_ae_dims_mean'])
    ax.fill_between(x_, results['accuracy_ae_dims_mean']+results['accuracy_ae_dims_std'], results['accuracy_ae_dims_mean']-results['accuracy_ae_dims_std'], alpha=0.2)

    ax.legend(['PCA', 'RNN-AE'], loc='upper right')
    ax.set(title='Classification accuracy', xlabel='Dimensions', ylabel='Accuracy')

    fig.tight_layout()
    fig.savefig(f"../results/{filename}_accuracy_dims_group.png")
    fig.clf()
    
    ### MSE violin plot ###
    fig, ax = plt.subplots()
    
    sns.violinplot(data=[
        results['mse_linear'],
        results['mse_ae']
    ],
                  ax=ax,
                   cut=0
                  )
    
    ax.set(
        ylabel='MSE',
        xticklabels=['PCA', 'RNN-AE']
          )
    
    fig.tight_layout()
    fig.savefig(f"../results/{filename}_mse_violin_group.png")
    fig.clf()
    
    ### Classifier performance ###
    fig, ax = plt.subplots()
    
    classifiers = ['LR', 'SVM', 'SGD', 'NB', 'RF']
    x_ = np.array([1, 2, 3, 4, 5])
    w = 0.4
    
    ax.bar(x_-w/2, results['accuracy_linear_mean'], w, yerr=results['accuracy_linear_std'], label='PCA')
    ax.bar(x_+w/2, results['accuracy_ae_mean'], w, yerr=results['accuracy_ae_std'], label='RNN-AE')
    
    ax.set(xlabel='Classifier', ylabel='Accuracy')
    ax.legend(loc='upper right')
    plt.xticks([1, 2, 3, 4, 5], classifiers)
   
    fig.tight_layout()
    fig.savefig(f"../results/{filename}_classifiers_group.png")
    fig.clf()
    
    ### Loss & accuracy vs. tasks ###
    fig, ax = plt.subplots(2, 1, sharex='col')
    num_tasks = np.arange(1, len(results['linear_interpolation_loss_tasknum_mean'])+1)*2
    
    ax[0].plot(num_tasks, results['linear_interpolation_loss_tasknum_mean'])
    ax[0].fill_between(num_tasks, results['linear_interpolation_loss_tasknum_mean']+results['linear_interpolation_loss_tasknum_std'], results['linear_interpolation_loss_tasknum_mean']-results['linear_interpolation_loss_tasknum_std'], alpha=0.2)
    
    ax[0].plot(num_tasks, results['ae_interpolation_loss_tasknum_mean'])
    ax[0].fill_between(num_tasks, results['ae_interpolation_loss_tasknum_mean']+results['ae_interpolation_loss_tasknum_std'], results['ae_interpolation_loss_tasknum_mean']-results['ae_interpolation_loss_tasknum_std'], alpha=0.2)
    
    ax[0].set(title='Reconstruction loss', ylabel='MSE')
    
    ax[1].plot(num_tasks, results['linear_interpolation_accuracy_tasknum_mean'])
    ax[1].fill_between(num_tasks, results['linear_interpolation_accuracy_tasknum_mean']+results['linear_interpolation_accuracy_tasknum_std'], results['linear_interpolation_accuracy_tasknum_mean']-results['linear_interpolation_accuracy_tasknum_std'], alpha=0.2)
    
    ax[1].plot(num_tasks, results['ae_interpolation_accuracy_tasknum_mean'])
    ax[1].fill_between(num_tasks, results['ae_interpolation_accuracy_tasknum_mean']+results['ae_interpolation_accuracy_tasknum_std'], results['ae_interpolation_accuracy_tasknum_mean']-results['ae_interpolation_accuracy_tasknum_std'], alpha=0.2)
    
    ax[1].set(title='Classification accuracy', ylabel='Accuracy', xlabel='Number of training tasks')
    
    fig.tight_layout()
    fig.savefig(f"../results/{filename}_lossacc_vstasks_group.png")
    fig.clf()
    
    ### Violin plot: MSE, interpolation & standard ###
    fig, ax = plt.subplots()
    
    mse_combined = np.concatenate((
    results['mse_linear'],
    results['mse_ae'],
    results['linear_interpolation_loss_mean'],
    results['ae_interpolation_loss_mean']
    ))
    
    model_type = np.concatenate((
    np.zeros(len(results['mse_linear'])),
    np.ones(len(results['mse_ae'])),
    np.zeros(len(results['linear_interpolation_loss_mean'])),
    np.ones(len(results['ae_interpolation_loss_mean']))
    ))
    
    interp_type = np.concatenate((
    np.zeros(len(results['mse_linear'])),
    np.zeros(len(results['mse_ae'])),
    np.ones(len(results['linear_interpolation_loss_mean'])),
    np.ones(len(results['ae_interpolation_loss_mean']))
    ))
    
    df = pd.DataFrame([mse_combined, model_type, interp_type])
    df = df.T
    df.columns = ['MSE', 'Model', 'Interp']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'RNN-AE', inplace=True)
    df['Interp'].replace(0, 'Standard', inplace=True)
    df['Interp'].replace(1, 'Interpolated', inplace=True)
    
    sns.violinplot(data=df,
                  x='Interp',
                  y='MSE',
                  hue='Model',
                  ax=ax)
    
    ax.set(
    ylabel='MSE', xlabel='',
    xticklabels=['Standard', 'Interpolated']
    )
    
    ax.legend(loc='upper left')
    
    fig.tight_layout()
    fig.savefig(f"../results/{filename}_mse_interpvsstandard.png")
    fig.clf()
    
    ### Violin plot: MSE, transfer & individual
    # Need transfer data
    
# Get overall PCA plot
def overall_pca(data:dict):
    '''
    Get overall PCA curve for data
    
    Args:
        data (dict): data dict for grand-average data
        
    Produces plot of overall cumulative variance vs. components
    '''
    # Combine train & test data
    X = np.concatenate((data['X_train'], data['X_test']), axis=0)
    
    # Reshape data
    X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    
    # Get PCA model
    pca = PCA()
    
    # Get cumulative variance
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    x_ = np.arange(1, len(cumsum)+1)
    
    # Plot explained variance
    fig, ax = plt.subplots()
    ax.plot(x_, cumsum)
    
    ax.set(xlabel='Dimensions', ylabel='Variance explained')
    
    fig.tight_layout()
    fig.savefig("../results/grandaverage_explainedvariance.png")
    fig.clf()

# Create definitive plots
def generate_plots(individual, transfer, grandaverage, supplementary):
    '''
    Generates figures for manuscript
    
    Args:
        individual: dict of individual results
        transfer: dict of transfer results
        grandaverage: dict of grandaverage results
        supplementary: dict of miscellaneous results for plots (examples missing from full analysis, etc.)
    '''
    ### Set plot style ###
    sns.set_style('ticks')
    palette = sns.color_palette('Greys_r', 4)
    sns.set_palette(palette)
    
    colour_sae = 'k'
    colour_pca = 'darkgrey'
    
    ### 1A: task grid ###
        # Produced externally (grid of photos)
        
    ### 1B: variance explained ###
        # grandaverage - explained_variance_linear_dims, explained_variance_ae_dims
            # supplementary - pca -> explained_variance_ratio_
                # Can try again w R2 etc. for PCA & AE for comparison if necessary
    
    # Temporary X axis for plotting - every second dimension used
    #x_ = (np.arange(1, len(grandaverage['explained_variance_linear_dims'])+1)*2)-1
    pca = supplementary['pca']
    variance_explained = np.cumsum(pca.explained_variance_ratio_)
    x = np.arange(1, len(variance_explained)+1)
    
    fig, ax = plt.subplots(figsize=(9,4.5))
    
    ax.plot(x, variance_explained)
    
    #ax.plot(x_, grandaverage['explained_variance_linear_dims'], label='PCA')
    #ax.plot(x_, grandaverage['explained_variance_ae_dims'], label='RNN-AE')
    
    ax.set(
        xlabel = 'Dimensions',
        ylabel = 'Variance explained (%)',
        ylim = [0, 1],
        xticks = np.arange(1, 22, 4)
    )
    
    #ax.legend()
    
    sns.despine(ax=ax, offset=10, trim=True)
    
    fig.tight_layout()
    saveFig(fig, '1B')
    
    ### 1C: PC weights ###
        # supplementary - pca_weights
    weights = supplementary['pca_weights']
    x = np.arange(0, weights.shape[1])
    w = 0.2
    xlabels = [
        'Wrist X',
        'Wrist Y',
        'Thumb X',
        'Thumb Y',
        'X acceleration',
        'Y acceleration',
        'Z acceleration',
        'Thumb MCPJ',
        'Thumb IPJ',
        'Index MCPJ',
        'Index DIPJ',
        'Middle MCPJ',
        'Middle DIPJ',
        'Ring MCPJ',
        'Ring DIPJ',
        'Little MCPJ',
        'Little DIPJ',
        'Rotation L/R',
        'Position L/R',
        'Rotation U/D',
        'Position U/D'
    ]
    
    fig, ax = plt.subplots(figsize=(20,5))
    
    ax.bar(
        x - w,
        weights[0, :],
        w,
        label = 'PC 1',
        color = 'k'
          )
    ax.bar(
        x,
        weights[1, :],
        w,
        label = 'PC 2',
        color = 'grey'
          )
    ax.bar(
        x + w,
        weights[2, :],
        w,
        label = 'PC 3',
        color = 'darkgrey'
          )
    
    ax.set(
        xlabel = 'Kinematic measures',
        ylabel = 'Weights',
        #xticks = np.arange(0, weights.shape[1]),
        #xticklabels = np.arange(1, weights.shape[1]+1) 
    )
    
    ax.legend(frameon=False, loc='upper right')
    
    ax.plot([-0.2, 20.3], [0, 0], color='k', linestyle='dotted')
    
    sns.despine(ax=ax, offset=10, trim=True)
    plt.xticks(x, xlabels, rotation=45, ha='right')
    fig.tight_layout()
    saveFig(fig, '1C')
    
    ### 2A: Violin, MSE; PCA, AE, ind, tf - direct ###
        # individual - mse_linear, mse_ae
        # transfer - mse_linear, mse_ae
    fig, ax = plt.subplots()
    
    mse_combined = np.concatenate((
        individual['mse_linear'],
        individual['mse_ae'],
        transfer['mse_linear'],
        transfer['mse_ae']
    ))
    
    model_type = np.concatenate((
        np.zeros(len(individual['mse_linear'])),
        np.ones(len(individual['mse_ae'])),
        np.zeros(len(transfer['mse_linear'])),
        np.ones(len(transfer['mse_ae']))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(individual['mse_linear'])),
        np.zeros(len(individual['mse_ae'])),
        np.ones(len(transfer['mse_linear'])),
        np.ones(len(transfer['mse_ae']))
    ))
    
    jitter = 0.1
    x_pos = np.concatenate((
        np.zeros(len(individual['mse_linear'])) + np.random.standard_normal(len(individual['mse_linear']))*jitter,
        np.ones(len(individual['mse_ae']))*0.5 + np.random.standard_normal(len(individual['mse_ae']))*jitter,
        np.ones(len(transfer['mse_linear']))*2.5 + np.random.standard_normal(len(individual['mse_linear']))*jitter,
        np.ones(len(transfer['mse_ae']))*3 + np.random.standard_normal(len(individual['mse_ae']))*jitter
    ))
    
    df = pd.DataFrame([mse_combined, model_type, data_source, x_pos])
    df = df.T
    df.columns = ['MSE', 'Model', 'Source', 'X Position']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'SAE', inplace=True)
    df['Source'].replace(0, 'Individual', inplace=True)
    df['Source'].replace(1, 'Transfer', inplace=True)
    
    sns.scatterplot(
        data = df,
        x = 'X Position',
        y = 'MSE',
        hue = 'Model',
        hue_order = ['PCA', 'SAE'],
        style = 'Model',
        markers = ['o', 'X'],
        palette = {'PCA': colour_pca, 'SAE': colour_sae},
        x_jitter = 0.2,
        ax = ax
    )
    
    ax.set(
        ylabel = 'MSE',
        xlabel = '',
        xticks = [0.25, 2.75],
        xticklabels = ['Individual', 'Transfer'],
        ylim = [0, 0.015],
        yticks = np.arange(0, 0.016, 0.005)
    )
    
    # Create custom legend
    custom = [
        Line2D([], [], marker='o', color=colour_pca, linestyle='None'),
        Line2D([], [], marker='X', color=colour_sae, linestyle='None'),
    ]
    
    ax.legend(custom, ['PCA', 'SAE'], loc='upper center', ncol=2, bbox_to_anchor=(0,0,1,1.3), frameon=False)
    
    sns.despine(ax=ax, offset=10, trim=True)
    
    # Add annotations for outliers
    '''
    ax.annotate(
        '',
        xy = (2.2, 0.015),
        xytext = (2.2, 0.013),
        arrowprops = {
            'facecolor': 'k',
            'arrowstyle': 'simple',
        }
    )
    ax.text(
        2.15, 0.0135,
        '0.045',
        color = 'k',
        ha = 'right', va = 'baseline',
        size = 12,
        #rotation = 90,
    )
    
    ax.annotate(
        '',
        xy = (3.2, 0.015),
        xytext = (3.2, 0.013),
        arrowprops = {
            'facecolor': 'darkgrey',
            'arrowstyle': 'simple',
        }
    )
    ax.text(
        3.25, 0.0135,
        '0.025',
        color = 'darkgrey',
        ha = 'left', va = 'baseline',
        size = 12,
        #rotation = 90,
    )
    '''
    
    fig.tight_layout()
    saveFig(fig, '2A')
    
    ### 2B: MSE vs. dims, PCA & AE, ind, direct ###
        # individual - mse_linear_dims_mean/std, mse_ae_dims_mean/std
    fig, ax = plt.subplots(figsize=(9,3))
    
    x = (np.arange(1, len(individual['mse_linear_dims_mean'])+1)*2)-1
    
    ax.plot(x, individual['mse_linear_dims_mean'], color=colour_pca)
    ax.fill_between(x,
                   individual['mse_linear_dims_mean'] + individual['mse_linear_dims_std'],
                    individual['mse_linear_dims_mean'] - individual['mse_linear_dims_std'],
                    color=colour_pca,
                    alpha = 0.2
                   )
    
    ax.plot(x, individual['mse_ae_dims_mean'], color=colour_sae, linestyle='solid')
    ax.fill_between(x,
                   individual['mse_ae_dims_mean'] + individual['mse_ae_dims_std'],
                    individual['mse_ae_dims_mean'] - individual['mse_ae_dims_std'],
                    color=colour_sae,
                    alpha = 0.2
                   )
    
    ax.legend(['PCA', 'SAE'], loc='upper right', frameon=False)
    ax.set(
        xlabel = 'Dimensions',
        ylabel = 'MSE',
        yticks = [0, 0.02, 0.04],
        xticks = np.arange(1, 22, 4)
    )
    
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, '2B')
    
    ### 2C, D, E, F: examples ###
        # Figure out trials to use in NB
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10,10))
    
    n = 37
    ch = 6
    
    x = np.linspace(0, 2, num=256)
    
    # Individual, direct: PCA + AE
    ax[0,0].plot(x, supplementary['actual'][n, :, ch], 'k--')
    ax[0,0].plot(x, supplementary['individual_linear_direct'][n, :, ch])
    ax[0,0].plot(x, supplementary['individual_ae_direct'][n, :, ch])
    
    ax[0,0].legend(['Actual', 'PCA', 'SAE'], loc='upper right')
    ax[0,0].set(
        title = 'Individual',
        ylabel = 'Amplitude (a.u.)'
    )
   
    # Transfer, direct: PCA + AE
    ax[0,1].plot(x, supplementary['actual'][n, :, ch], 'k--')
    ax[0,1].plot(x, supplementary['transfer_linear_direct'][n, :, ch])
    ax[0,1].plot(x, supplementary['transfer_ae_direct'][n, :, ch])
    
    ax[0,1].legend(['Actual', 'PCA', 'SAE'], loc='upper right')
    ax[0,1].set(
        title = 'Transfer',
    )
   
    # PCA, direct: individual + transfer
    ax[1,0].plot(x, supplementary['actual'][n, :, ch], 'k--')
    ax[1,0].plot(x, supplementary['individual_linear_direct'][n, :, ch])
    ax[1,0].plot(x, supplementary['transfer_linear_direct'][n, :, ch])
    
    ax[1,0].legend(['Actual', 'Individual', 'Transfer'], loc='upper right')
    ax[1,0].set(
        title = 'PCA',
        ylabel = 'Amplitude (a.u.)',
        xlabel = 'Time (s)'
    )
   
    # AE, direct: individual + transfer
    ax[1,1].plot(x, supplementary['actual'][n, :, ch], 'k--')
    ax[1,1].plot(x, supplementary['individual_ae_direct'][n, :, ch])
    ax[1,1].plot(x, supplementary['transfer_ae_direct'][n, :, ch])
    
    ax[1,1].legend(['Actual', 'Individual', 'Transfer'], loc='upper right')
    ax[1,1].set(
        title = 'SAE',
        xlabel = 'Time (s)'
    )
    
    fig.tight_layout()
    sns.despine(ax=ax[0,0], offset=10, trim=True)
    sns.despine(ax=ax[0,1], offset=10, trim=True)
    sns.despine(ax=ax[1,0], offset=10, trim=True)
    sns.despine(ax=ax[1,1], offset=10, trim=True)
    #saveFig(fig, '2C')
    
    ### 2C: examples, single plot ###
    fig, ax = plt.subplots(figsize=(8,8))
    
    n = 37
    ch = 6
    
    x = np.linspace(0, 2, num=256)
    
    lw = 3
    ax.plot(x, supplementary['actual'][n, :, ch], color='grey', linestyle='dotted', linewidth=lw)
    ax.plot(x, supplementary['individual_linear_direct'][n, :, ch], color=colour_pca, linestyle='solid', linewidth=lw)
    ax.plot(x, supplementary['individual_ae_direct'][n, :, ch], color=colour_sae, linestyle='solid', linewidth=lw)
    ax.plot(x, supplementary['transfer_linear_direct'][n, :, ch], color=colour_pca, linestyle='dashed', linewidth=lw)
    ax.plot(x, supplementary['transfer_ae_direct'][n, :, ch], color=colour_sae, linestyle='dashed', linewidth=lw)
    
    ax.legend(['Actual', 'Individual PCA', 'Individual SAE', 'Transfer PCA', 'Transfer SAE'], frameon=False, loc='upper right')
    
    ax.set(
        ylabel = 'Amplitude (a.u.)',
        xlabel = 'Time (s)',
        ylim = [0, 1],
        xticks = np.arange(0, 2.1, 0.5)
    )
    
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, '2C')
    
    ### 3A: Violin, accuracy; PCA, AE, ind, tf - direct ###
        # individual - accuracy_linear_max, accuracy_ae_max? - check
        # transfer - accuracy_linear_max, accuracy_ae_max
            # Take either max or first element (LR) from each entry in list
            # result = [r[0] for r in list]
            
    individual_linear = [r[0] for r in individual['accuracy_linear_max']]
    individual_ae = [r[0] for r in individual['accuracy_ae_max']]
    transfer_linear = [r[0] for r in transfer['accuracy_linear_max']]
    transfer_ae = [r[0] for r in transfer['accuracy_ae_max']]
    
    fig, ax = plt.subplots()
    
    accuracy_combined = np.concatenate((
        individual_linear,
        individual_ae,
        transfer_linear,
        transfer_ae
    ))
    
    model_type = np.concatenate((
        np.zeros(len(individual_linear)),
        np.ones(len(individual_ae)),
        np.zeros(len(transfer_linear)),
        np.ones(len(transfer_ae))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(individual_linear)),
        np.zeros(len(individual_ae)),
        np.ones(len(transfer_linear)),
        np.ones(len(transfer_ae))
    ))
    
    jitter = 0.1
    x_pos = np.concatenate((
        np.zeros(len(individual_linear)) + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(individual_ae))*0.5 + np.random.standard_normal(len(individual_ae))*jitter,
        np.ones(len(transfer_linear))*2.5 + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(transfer_ae))*3 + np.random.standard_normal(len(individual_ae))*jitter
    ))
    
    df = pd.DataFrame([accuracy_combined, model_type, data_source, x_pos])
    df = df.T
    df.columns = ['Accuracy', 'Model', 'Source', 'X Position']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'SAE', inplace=True)
    df['Source'].replace(0, 'Individual', inplace=True)
    df['Source'].replace(1, 'Transfer', inplace=True)
    
    sns.scatterplot(
        data = df,
        x = 'X Position',
        y = 'Accuracy',
        hue = 'Model',
        hue_order = ['PCA', 'SAE'],
        style = 'Model',
        markers = ['o', 'X'],
        palette = {'PCA': colour_pca, 'SAE': colour_sae},
        ax = ax
    )
    
    '''sns.violinplot(
        data = df,
        x = 'Source',
        y = 'Accuracy',
        hue = 'Model',
        ax = ax,
        cut = 0
    )'''
    
    ax.set(
        ylabel = 'Accuracy',
        xlabel = '',
        xticks = [0.25, 2.75],
        xticklabels = ['Individual', 'Transfer'],
        ylim = [0, 1]
    )
    ax.axhline(y=0.91, color='grey', linestyle='dotted', label='Baseline')
    
    # Create custom legend
    custom = [
        Line2D([], [], marker='o', color=colour_pca, linestyle='None'),
        Line2D([], [], marker='X', color=colour_sae, linestyle='None'),
        Line2D([], [], color='grey', linestyle='dotted')
    ]
    
    ax.legend(custom, ['PCA', 'SAE', 'Baseline'], loc='upper center', ncol=3, bbox_to_anchor=(0,0,1,1.3), frameon=False)
    
#    ax.legend(['PCA', 'SAE', 'Baseline'], loc='upper center', frameon=False)
    
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, '3A')
    
    ### 3B: Accuracy per model, PCA, AE, ind, direct ###
        # individual - accuracy_linear, accuracy_ae
        # accuracy_linear is same as accuracy_linear_max it seems
    classifiers = ['LR', 'SVM', 'SGD', 'NB', 'RF']
    x = np.array([1, 2, 3, 4, 5])
    w = 0.4
    
    fig, ax = plt.subplots()
    
    ax.bar(
          x-w/2,
          individual['accuracy_linear_mean'],
          w,
          yerr = individual['accuracy_linear_std'],
          label = 'PCA',
          color = colour_pca
          )
    ax.bar(
          x+w/2,
          individual['accuracy_ae_mean'],
          w,
          yerr = individual['accuracy_ae_std'],
          label = 'SAE',
          color = colour_sae
          )
    
    ax.axhline(y=0.91, color='grey', linestyle='dotted', label='Baseline')
    
    ax.set(
        xlabel = 'Classifier',
        ylabel = 'Accuracy',
        ylim = [0, 1]
    )
    
    ax.legend(loc='upper center', frameon=False, ncol=3, bbox_to_anchor=(0,0,1,1.3))
    sns.despine(ax=ax, offset=10, trim=True)
    plt.xticks(x, classifiers, rotation=45)
    fig.tight_layout()
    saveFig(fig, '3B')
    
    ### 3C: Accuracy vs. dims, PCA & AE, ind, direct ###
        # individual - accuracy_linear/ae_dims_mean/std
    x = (np.arange(1, len(individual['accuracy_linear_dims_mean'])+1)*2)-1
    
    fig, ax = plt.subplots()
    
    ax.plot(x, individual['accuracy_linear_dims_mean'], color=colour_pca)
    ax.fill_between(x,
                   individual['accuracy_linear_dims_mean'] + individual['accuracy_linear_dims_std'],
                   individual['accuracy_linear_dims_mean'] - individual['accuracy_linear_dims_std'],
                   alpha = 0.2,
                   color = colour_pca
                   )
    ax.plot(x, individual['accuracy_ae_dims_mean'], color=colour_sae, linestyle='solid')
    ax.fill_between(x,
                   individual['accuracy_ae_dims_mean'] + individual['accuracy_ae_dims_std'],
                   individual['accuracy_ae_dims_mean'] - individual['accuracy_ae_dims_std'],
                   alpha = 0.2,
                   color=colour_sae
                   )
    ax.axhline(y=0.91, color='grey', linestyle='dotted', label='Baseline')
    
    ax.legend(['PCA', 'SAE', 'Baseline'], loc='lower right', frameon=False)
    ax.set(
        xlabel = 'Dimensions',
        ylabel = 'Accuracy',
        ylim = [0, 1],
        xticks = np.arange(1, 22, 4)
    )
   
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, '3C')
    
    ### 4A: Violin, interpolation; PCA, AE, ind, tf ###
        # individual - linear_interpolation_loss_mean, ae_interpolation_loss_mean
        # transfer - linear_interpolation_loss_mean, ae_interpolation_loss_mean
    individual_linear = individual['linear_interpolation_loss_mean']
    individual_ae = individual['ae_interpolation_loss_mean']
    transfer_linear = transfer['linear_interpolation_loss_mean']
    transfer_ae = transfer['ae_interpolation_loss_mean']
    
    fig, ax = plt.subplots()
    
    interpolation_combined = np.concatenate((
        individual_linear,
        individual_ae,
        transfer_linear,
        transfer_ae
    ))
    
    model_type = np.concatenate((
        np.zeros(len(individual_linear)),
        np.ones(len(individual_ae)),
        np.zeros(len(transfer_linear)),
        np.ones(len(transfer_ae))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(individual_linear)),
        np.zeros(len(individual_ae)),
        np.ones(len(transfer_linear)),
        np.ones(len(transfer_ae))
    ))
    
    jitter = 0.1
    x_pos = np.concatenate((
        np.zeros(len(individual_linear)) + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(individual_ae))*0.5 + np.random.standard_normal(len(individual_ae))*jitter,
        np.ones(len(transfer_linear))*2.5 + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(transfer_ae))*3 + np.random.standard_normal(len(individual_ae))*jitter
    ))
    
    df = pd.DataFrame([interpolation_combined, model_type, data_source, x_pos])
    df = df.T
    df.columns = ['Interpolation', 'Model', 'Source', 'X Position']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'SAE', inplace=True)
    df['Source'].replace(0, 'Individual', inplace=True)
    df['Source'].replace(1, 'Transfer', inplace=True)
    
    sns.scatterplot(
        data = df,
        x = 'X Position',
        y = 'Interpolation',
        hue = 'Model',
        hue_order = ['PCA', 'SAE'],
        style = 'Model',
        markers = ['o', 'X'],
        palette = {'PCA': colour_pca, 'SAE': colour_sae},
        ax = ax
    )
    
    '''sns.violinplot(
        data = df,
        x = 'Source',
        y = 'Interpolation',
        hue = 'Model',
        ax = ax,
        cut = 0
    )'''
    
    ax.set(
        ylabel = 'MSE',
        xlabel = '',
        xticks = [0.25, 2.75],
        xticklabels = ['Individual', 'Transfer'],
        ylim = [0, 0.015],
        yticks = np.arange(0, 0.016, 0.005)
    )
    
    # Create custom legend
    custom = [
        Line2D([], [], marker='o', color=colour_pca, linestyle='None'),
        Line2D([], [], marker='X', color=colour_sae, linestyle='None')
    ]
    
    ax.legend(custom, ['PCA', 'SAE'], loc='upper center', ncol=2, bbox_to_anchor=(0,0,1,1.3), frameon=False)
    
    '''
    # Add annotations for outliers
    ax.annotate(
        '',
        xy = (2.2, 0.015),
        xytext = (2.2, 0.013),
        arrowprops = {
            'facecolor': 'k',
            'arrowstyle': 'simple',
        }
    )
    ax.text(
        2.15, 0.0135,
        '0.047',
        color = 'k',
        ha = 'right', va = 'baseline',
        size = 12,
        #rotation = 90,
    )
    
    ax.annotate(
        '',
        xy = (3.2, 0.015),
        xytext = (3.2, 0.013),
        arrowprops = {
            'facecolor': 'darkgrey',
            'arrowstyle': 'simple',
        }
    )
    ax.text(
        3.25, 0.0135,
        '0.033',
        color = 'darkgrey',
        ha = 'left', va = 'baseline',
        size = 12,
        #rotation = 90,
    )
    '''
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, '4A')
    
    ### 4B: Interp loss for each movement; PCA & AE, ind ###
        # individual - linear/ae_interpolation_loss
        # OR: linear/ae_ranked_tasks + linear/ae_ranked_loss_mean/std
            # For showing in ranked order
            # Need COMMON order to compare methods
            
        # Replacing with MSE, direct vs interp for ind & tf
    fig, ax = plt.subplots(1,2, sharey='row', figsize=(10,5))
    
    direct_linear = individual['mse_linear']
    direct_ae = individual['mse_ae']
    interpolated_linear = individual['linear_interpolation_loss_mean']
    interpolated_ae = individual['ae_interpolation_loss_mean']
    
    interpolation_combined = np.concatenate((
        direct_linear,
        direct_ae,
        interpolated_linear,
        interpolated_ae
    ))
    
    model_type = np.concatenate((
        np.zeros(len(direct_linear)),
        np.ones(len(direct_ae)),
        np.zeros(len(interpolated_linear)),
        np.ones(len(interpolated_ae))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(direct_linear)),
        np.zeros(len(direct_ae)),
        np.ones(len(interpolated_linear)),
        np.ones(len(interpolated_ae))
    ))
    
    jitter = 0.1
    x_pos = np.concatenate((
        np.zeros(len(individual_linear)) + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(individual_ae))*0.5 + np.random.standard_normal(len(individual_ae))*jitter,
        np.ones(len(transfer_linear))*2.5 + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(transfer_ae))*3 + np.random.standard_normal(len(individual_ae))*jitter
    ))
    
    df = pd.DataFrame([interpolation_combined, model_type, data_source, x_pos])
    df = df.T
    df.columns = ['Interpolation', 'Model', 'Interp', 'X Position']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'SAE', inplace=True)
    df['Interp'].replace(0, 'Direct', inplace=True)
    df['Interp'].replace(1, 'Interpolated', inplace=True)
    
    sns.scatterplot(
        data = df,
        x = 'X Position',
        y = 'Interpolation',
        hue = 'Model',
        hue_order = ['PCA', 'SAE'],
        style = 'Model',
        markers = ['o', 'X'],
        palette = {'PCA': colour_pca, 'SAE': colour_sae},
        ax = ax[0]
    )
    '''sns.violinplot(
        data = df,
        x = 'Interp',
        y = 'Interpolation',
        hue = 'Model',
        ax = ax[0],
        cut = 0
    )'''
    
    ax[0].set(
        title = 'Individual',
        ylabel = 'MSE',
        xlabel = '',
        xticks = [0.25, 2.75],
        xticklabels = ['Direct', 'Interpolated'],
        ylim = [0, 0.015],
        yticks = np.arange(0, 0.016, 0.005)
    )
    
    # Create custom legend
    custom = [
        Line2D([], [], marker='o', color=colour_pca, linestyle='None'),
    ]
    
    ax[0].legend(custom, ['PCA'], loc='upper right', ncol=2, bbox_to_anchor=(0,0,1,1.3), frameon=False)
    
    direct_linear = transfer['mse_linear']
    direct_ae = transfer['mse_ae']
    interpolated_linear = transfer['linear_interpolation_loss_mean']
    interpolated_ae = transfer['ae_interpolation_loss_mean']
    
    interpolation_combined = np.concatenate((
        direct_linear,
        direct_ae,
        interpolated_linear,
        interpolated_ae
    ))
    
    model_type = np.concatenate((
        np.zeros(len(direct_linear)),
        np.ones(len(direct_ae)),
        np.zeros(len(interpolated_linear)),
        np.ones(len(interpolated_ae))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(direct_linear)),
        np.zeros(len(direct_ae)),
        np.ones(len(interpolated_linear)),
        np.ones(len(interpolated_ae))
    ))
    
    jitter = 0.1
    x_pos = np.concatenate((
        np.zeros(len(individual_linear)) + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(individual_ae))*0.5 + np.random.standard_normal(len(individual_ae))*jitter,
        np.ones(len(transfer_linear))*2.5 + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(transfer_ae))*3 + np.random.standard_normal(len(individual_ae))*jitter
    ))
    
    df = pd.DataFrame([interpolation_combined, model_type, data_source, x_pos])
    df = df.T
    df.columns = ['Interpolation', 'Model', 'Interp', 'X Position']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'SAE', inplace=True)
    df['Interp'].replace(0, 'Direct', inplace=True)
    df['Interp'].replace(1, 'Interpolated', inplace=True)
    
    sns.scatterplot(
        data = df,
        x = 'X Position',
        y = 'Interpolation',
        hue = 'Model',
        hue_order = ['PCA', 'SAE'],
        style = 'Model',
        markers = ['o', 'X'],
        palette = {'PCA': colour_pca, 'SAE': colour_sae},
        ax = ax[1]
    )
    '''sns.violinplot(
        data = df,
        x = 'Interp',
        y = 'Interpolation',
        hue = 'Model',
        ax = ax[1],
        cut = 0
    )'''
    
    ax[1].set(
        title = 'Transfer',
        xlabel = '',
        ylabel = '',
        xticks = [0.25, 2.75],
        xticklabels = ['Direct', 'Interpolated'],
        ylim = [0, 0.016],
        yticks = np.arange(0, 0.016, 0.005)
    )
    ax[1].get_legend().remove()
    
    # Create custom legend
    custom = [
        Line2D([], [], marker='X', color=colour_sae, linestyle='None')
    ]
    
    ax[1].legend(custom, ['SAE'], loc='upper left', ncol=2, bbox_to_anchor=(0,0,1,1.3), frameon=False)
    
    sns.despine(ax=ax[0], offset=10, trim=True)
    sns.despine(ax=ax[1], offset=10, trim=True)
    ax[1].axes.get_yaxis().set_visible(False)
    ax[1].spines['left'].set_visible(False)
    
    '''
    # Add annotations for outliers
    ax[1].annotate(
        '',
        xy = (2.2, 0.015),
        xytext = (2.2, 0.013),
        arrowprops = {
            'facecolor': 'k',
            'arrowstyle': 'simple',
        }
    )
    ax[1].text(
        2.15, 0.0135,
        '0.047',
        color = 'k',
        ha = 'right', va = 'baseline',
        size = 12,
        #rotation = 90,
    )
    
    ax[1].annotate(
        '',
        xy = (3.2, 0.015),
        xytext = (3.2, 0.013),
        arrowprops = {
            'facecolor': 'darkgrey',
            'arrowstyle': 'simple',
        }
    )
    ax[1].text(
        3.25, 0.0135,
        '0.033',
        color = 'darkgrey',
        ha = 'left', va = 'baseline',
        size = 12,
        #rotation = 90,
    )
    
    ax[1].annotate(
        '',
        xy = (-0.3, 0.015),
        xytext = (-0.3, 0.013),
        arrowprops = {
            'facecolor': 'k',
            'arrowstyle': 'simple',
        }
    )
    ax[1].text(
        -0.35, 0.0135,
        '0.045',
        color = 'k',
        ha = 'right', va = 'baseline',
        size = 12,
        #rotation = 90,
    )
    
    ax[1].annotate(
        '',
        xy = (0.7, 0.015),
        xytext = (0.7, 0.013),
        arrowprops = {
            'facecolor': 'darkgrey',
            'arrowstyle': 'simple',
        }
    )
    ax[1].text(
        0.75, 0.0135,
        '0.025',
        color = 'darkgrey',
        ha = 'left', va = 'baseline',
        size = 12,
        #rotation = 90,
    )
    '''
    fig.tight_layout()
    saveFig(fig, '4B')
    
    ### 4C, D, E, F: examples ###
        # Use NB to determine examples to use
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10,10))
    
    n = 54
    ch = 17
    
    x = np.linspace(0, 2, num=256)
    
    # PCA, individual: direct + interpolated
    ax[0,0].plot(x, supplementary['actual'][n, :, ch], 'k--')
    ax[0,0].plot(x, supplementary['individual_linear_direct'][n, :, ch])
    ax[0,0].plot(x, supplementary['individual_linear_interpolated'][n, :, ch])
    
    ax[0,0].legend(['Actual', 'Direct', 'Interpolated'], loc='upper right')
    ax[0,0].set(
        title = 'Individual, PCA',
        ylabel = 'Amplitude (a.u.)'
    )
   
    # AE, individual: direct + interpolated
    ax[0,1].plot(x, supplementary['actual'][n, :, ch], 'k--')
    ax[0,1].plot(x, supplementary['individual_ae_direct'][n, :, ch])
    ax[0,1].plot(x, supplementary['individual_ae_interpolated'][n, :, ch])
    
    ax[0,1].legend(['Actual', 'Direct', 'Interpolated'], loc='upper right')
    ax[0,1].set(
        title = 'Individual, RNN-AE',
    )
   
    # PCA, transfer: direct + interpolated
    ax[1,0].plot(x, supplementary['actual'][n, :, ch], 'k--')
    ax[1,0].plot(x, supplementary['transfer_linear_direct'][n, :, ch])
    ax[1,0].plot(x, supplementary['transfer_linear_interpolated'][n, :, ch])
    
    ax[1,0].legend(['Actual', 'Direct', 'Interpolated'], loc='upper right')
    ax[1,0].set(
        title = 'Transfer, PCA',
        ylabel = 'Amplitude (a.u.)',
        xlabel = 'Time (s)'
    )
   
    # AE, transfer: direct + interpolated
    ax[1,1].plot(x, supplementary['actual'][n, :, ch], 'k--')
    ax[1,1].plot(x, supplementary['transfer_ae_direct'][n, :, ch])
    ax[1,1].plot(x, supplementary['transfer_ae_interpolated'][n, :, ch])
    
    ax[1,1].legend(['Actual', 'Direct', 'Interpolated'], loc='upper right')
    ax[1,1].set(
        title = 'Transfer, RNN-AE',
        xlabel = 'Time (s)'
    )
    
    fig.tight_layout()
    sns.despine(ax=ax[0,0], offset=10, trim=True)
    sns.despine(ax=ax[0,1], offset=10, trim=True)
    sns.despine(ax=ax[1,0], offset=10, trim=True)
    sns.despine(ax=ax[1,1], offset=10, trim=True)
   # saveFig(fig, '4C')

    ### 4C: examples, single plot ###
    fig, ax = plt.subplots(2, 1, sharex='col', figsize=(8,10))
    
    n = 54
    ch = 17
    
    x = np.linspace(0, 2, num=256)
    
    lw = 3
    ax[0].plot(x, supplementary['actual'][n, :, ch], color='grey', linestyle='dotted', linewidth=lw)
    
    ax[0].plot(x, supplementary['individual_linear_direct'][n, :, ch], color=colour_pca, linestyle='solid', linewidth=lw)
    ax[0].plot(x, supplementary['individual_ae_direct'][n, :, ch], color=colour_sae, linestyle='solid', linewidth=lw)
    ax[0].plot(x, supplementary['transfer_linear_direct'][n, :, ch], color=colour_pca, linestyle='dashed', linewidth=lw)
    ax[0].plot(x, supplementary['transfer_ae_direct'][n, :, ch], color=colour_sae, linestyle='dashed', linewidth=lw)
    
    ax[1].plot(x, supplementary['actual'][n, :, ch], color='grey', linestyle='dotted', linewidth=lw)
    
    ax[1].plot(x, supplementary['individual_linear_interpolated'][n, :, ch], color=colour_pca, linestyle='solid', linewidth=lw)
    ax[1].plot(x, supplementary['individual_ae_interpolated'][n, :, ch], color=colour_sae, linestyle='solid', linewidth=lw)
    ax[1].plot(x, supplementary['transfer_linear_interpolated'][n, :, ch], color=colour_pca, linestyle='dashed', linewidth=lw)
    ax[1].plot(x, supplementary['transfer_ae_interpolated'][n, :, ch], color=colour_sae, linestyle='dashed', linewidth=lw)
    
    ax[1].legend(['Actual', 'Individual PCA', 'Individual SAE', 'Transfer PCA', 'Transfer SAE'], frameon=False, loc='lower left')
    
    ax[0].set(
        title = 'Direct',
        ylabel = 'Amplitude (a.u.)',
        #xlabel = 'Time (s)',
        ylim = [0, 1],
        xticks = np.arange(0, 2.1, 0.5)
    )
    ax[1].set(
        title = 'Interpolated',
        ylabel = 'Amplitude (a.u.)',
        xlabel = 'Time (s)',
        ylim = [0, 1],
        xticks = np.arange(0, 2.1, 0.5)
    )
    
    fig.tight_layout()
    sns.despine(ax=ax[0], offset=10, trim=True)
    sns.despine(ax=ax[1], offset=10, trim=True)
    
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    
    saveFig(fig, '4C')
    
    
    ### 5A: MSE vs. tasks, PCA & AE, ind ###
        # individual - linear/ae_interpolation_loss_tasknum_mean/std
    fig, ax = plt.subplots()
    
    x = np.arange(1, len(individual['linear_interpolation_loss_tasknum_mean'])+1)*2
    
    ax.plot(
        x,
        individual['linear_interpolation_loss_tasknum_mean'],
        color=colour_pca
           )
    ax.fill_between(
        x,
        individual['linear_interpolation_loss_tasknum_mean'] + individual['linear_interpolation_loss_tasknum_std'],
        individual['linear_interpolation_loss_tasknum_mean'] - individual['linear_interpolation_loss_tasknum_std'],
        alpha = 0.2,
        color=colour_pca
    )
    
    ax.plot(
        x,
        individual['ae_interpolation_loss_tasknum_mean'],
        color=colour_sae,
        linestyle='solid'
           )
    ax.fill_between(
        x,
        individual['ae_interpolation_loss_tasknum_mean'] + individual['ae_interpolation_loss_tasknum_std'],
        individual['ae_interpolation_loss_tasknum_mean'] - individual['ae_interpolation_loss_tasknum_std'],
        alpha = 0.2,
        color=colour_sae
    )
    
    ax.set(
        xlabel = 'Number of training tasks',
        ylabel = 'MSE',
        xticks = np.arange(2, 15, 2),
        ylim = [0, 0.0125]
    )
   # ax.legend(['PCA', 'SAE'], loc='upper right', frameon=False)
    
    sns.despine(ax=ax, offset=10, trim=True)
    fig.tight_layout()
    saveFig(fig, '5A')
    
    ### 5B: Accuracy vs. tasks, PCA & AE, ind ###
        # individual - linear/ae_interpolation_accuracy_tasknum_mean/std
    fig, ax = plt.subplots()
    
    x = np.arange(1, len(individual['linear_interpolation_accuracy_tasknum_mean'])+1)*2
    
    ax.plot(
        x,
        individual['linear_interpolation_accuracy_tasknum_mean'],
        color=colour_pca
           )
    ax.fill_between(
        x,
        individual['linear_interpolation_accuracy_tasknum_mean'] + individual['linear_interpolation_accuracy_tasknum_std'],
        individual['linear_interpolation_accuracy_tasknum_mean'] - individual['linear_interpolation_accuracy_tasknum_std'],
        alpha = 0.2,
        color=colour_pca
    )
    
    ax.plot(
        x,
        individual['ae_interpolation_accuracy_tasknum_mean'],
        color=colour_sae,
        linestyle='solid'
           )
    ax.fill_between(
        x,
        individual['ae_interpolation_accuracy_tasknum_mean'] + individual['ae_interpolation_accuracy_tasknum_std'],
        individual['ae_interpolation_accuracy_tasknum_mean'] - individual['ae_interpolation_accuracy_tasknum_std'],
        alpha = 0.2,
        color=colour_sae
    )
    ax.axhline(y=0.91, color='grey', linestyle='dotted', label='Baseline')
    
    ax.set(
        xlabel = 'Number of training tasks',
        ylabel = 'Accuracy',
        ylim = [0, 1],
        xticks = np.arange(2, 15, 2)
    )
    ax.legend(['PCA', 'SAE', 'Baseline'], loc='lower right', frameon=False)
    
    sns.despine(ax=ax, offset=10, trim=True)
    fig.tight_layout()
    saveFig(fig, '5B')
    
    ### 6A: MSE vs. dims, tf & GA ###
        # transfer - mse_linear/ae_dims_mean/std
        # grandaverage - mse_linear/ae_dims
    fig, ax = plt.subplots()
    
    x = (np.arange(1, len(transfer['mse_linear_dims_mean'])+1)*2)-1
    
    ax.plot(x, transfer['mse_linear_dims_mean'], color=colour_pca, linestyle='dashed')
    ax.fill_between(x,
                   transfer['mse_linear_dims_mean'] + transfer['mse_linear_dims_std'],
                    transfer['mse_linear_dims_mean'] - transfer['mse_linear_dims_std'],
                    alpha = 0.2,
                    color=colour_pca
                   )
    
    ax.plot(x, transfer['mse_ae_dims_mean'], color=colour_sae, linestyle='dashed')
    ax.fill_between(x,
                   transfer['mse_ae_dims_mean'] + transfer['mse_ae_dims_std'],
                    transfer['mse_ae_dims_mean'] - transfer['mse_ae_dims_std'],
                    alpha = 0.2,
                    color=colour_sae
                   )
    
    ax.plot(x, grandaverage['mse_linear_dims'], color='grey', linestyle='dotted')
    ax.plot(x, grandaverage['mse_ae_dims'], color='grey', linestyle='dashdot')
    
    ax.legend(['PCA', 'SAE'], loc='upper right', ncol=2, frameon=False, bbox_to_anchor=(0,0,1,1.3))
    ax.set(
        xlabel = 'Dimensions',
        ylabel = 'MSE',
        xticks = np.arange(1, 22, 4),
        ylim = [0, 0.05]
    )
    
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, '6A')
    
    ### 6B: Accuracy vs. dims, tf & GA ###
        # transfer - accuracy_linear/ae_dims_mean/std
        # grandaverage - accuracy_linear/ae_dims
    fig, ax = plt.subplots()
    
    x = (np.arange(1, len(transfer['accuracy_linear_dims_mean'])+1)*2)-1
    
    ax.plot(x, transfer['accuracy_linear_dims_mean'], color=colour_pca, linestyle='dashed')
    ax.fill_between(x,
                   transfer['accuracy_linear_dims_mean'] + transfer['accuracy_linear_dims_std'],
                    transfer['accuracy_linear_dims_mean'] - transfer['accuracy_linear_dims_std'],
                    alpha = 0.2,
                    color=colour_pca
                   )
    
    ax.plot(x, transfer['accuracy_ae_dims_mean'], color=colour_sae, linestyle='dashed')
    ax.fill_between(x,
                   transfer['accuracy_ae_dims_mean'] + transfer['accuracy_ae_dims_std'],
                    transfer['accuracy_ae_dims_mean'] - transfer['accuracy_ae_dims_std'],
                    alpha = 0.2,
                    color=colour_sae
                   )
    
    ax.plot(x, grandaverage['accuracy_linear_dims'], color='grey', linestyle='dotted')
    ax.plot(x, grandaverage['accuracy_ae_dims'], color='grey', linestyle='dashdot')
    
    # Create custom legend
    custom = [
        Line2D([], [], color='grey', linestyle='dotted'),
        Line2D([], [], color='grey', linestyle='dashdot')
    ]
    ax.legend(custom, ['GA PCA', 'GA SAE'], loc='upper left', ncol=2, frameon=False, bbox_to_anchor=(0,0,1,1.3))
    ax.set(
        xlabel = 'Dimensions',
        ylabel = 'Accuracy',
        ylim = [0, 1],
        xticks = np.arange(1, 22, 4)
    )
    
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, '6B')
    
    ### 6C: MSE vs. tasks, tf & GA ###
        # transfer - linear/ae_interpolation_loss_tasknum_mean/std
        # grandaverage - linear/ae_interpolation_loss_tasknum
    fig, ax = plt.subplots()
    
    x = np.arange(1, len(transfer['linear_interpolation_loss_tasknum_mean'])+1)*2
    
    ax.plot(
        x,
        transfer['linear_interpolation_loss_tasknum_mean'],
        color=colour_pca,
        linestyle='dashed'
           )
    ax.fill_between(
        x,
        transfer['linear_interpolation_loss_tasknum_mean'] + transfer['linear_interpolation_loss_tasknum_std'],
        transfer['linear_interpolation_loss_tasknum_mean'] - transfer['linear_interpolation_loss_tasknum_std'],
        alpha = 0.2,
        color=colour_pca
    )
    
    ax.plot(
        x,
        transfer['ae_interpolation_loss_tasknum_mean'],
        color=colour_sae,
        linestyle='dashed'
           )
    ax.fill_between(
        x,
        transfer['ae_interpolation_loss_tasknum_mean'] + transfer['ae_interpolation_loss_tasknum_std'],
        transfer['ae_interpolation_loss_tasknum_mean'] - transfer['ae_interpolation_loss_tasknum_std'],
        alpha = 0.2,
        color=colour_sae
    )
    
    ax.plot(x, grandaverage['linear_interpolation_loss_tasknum'], color='grey', linestyle='dotted')
    ax.plot(x, grandaverage['ae_interpolation_loss_tasknum'], color='grey', linestyle='dashdot')
    
    ax.set(
        xlabel = 'Number of training tasks',
        ylabel = 'MSE',
        xticks = np.arange(2, 15, 2),
        ylim = [0, 0.03]
    )
    #ax.legend(['PCA', 'RNN-AE', 'GA PCA', 'GA RNN-AE'], loc='upper right')
    
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, '6C')
    
    ### 6D: Accuracy vs. tasks, tf & GA ###
        # transfer - linear/ae_interpolation_accuracy_tasknum_mean/std
        # grandaverage - linear/ae_interpolation_accuracy_tasknum
    fig, ax = plt.subplots()
    
    x = np.arange(1, len(transfer['linear_interpolation_accuracy_tasknum_mean'])+1)*2
    
    ax.plot(
        x,
        transfer['linear_interpolation_accuracy_tasknum_mean'],
        color=colour_pca,
        linestyle='dashed'
           )
    ax.fill_between(
        x,
        transfer['linear_interpolation_accuracy_tasknum_mean'] + transfer['linear_interpolation_accuracy_tasknum_std'],
        transfer['linear_interpolation_accuracy_tasknum_mean'] - transfer['linear_interpolation_accuracy_tasknum_std'],
        alpha = 0.2,
        color=colour_pca
    )
    
    ax.plot(
        x,
        transfer['ae_interpolation_accuracy_tasknum_mean'],
        color=colour_sae,
        linestyle='dashed'
           )
    ax.fill_between(
        x,
        transfer['ae_interpolation_accuracy_tasknum_mean'] + transfer['ae_interpolation_accuracy_tasknum_std'],
        transfer['ae_interpolation_accuracy_tasknum_mean'] - transfer['ae_interpolation_accuracy_tasknum_std'],
        alpha = 0.2,
        color=colour_sae
    )
    
    ax.plot(x, grandaverage['linear_interpolation_accuracy_tasknum'], color='grey', linestyle='dotted')
    ax.plot(x, grandaverage['ae_interpolation_accuracy_tasknum'], color='grey', linestyle='dashdot')
    
    ax.set(
        xlabel = 'Number of training tasks',
        ylabel = 'Accuracy',
        ylim = [0, 1],
        xticks = np.arange(2, 15, 2)
    )
    #ax.legend(['PCA', 'RNN-AE', 'GA PCA', 'GA RNN-AE'], loc='lower right')
    
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, '6D')
        
    # Violin plots for interp v direct recon for ind & tf?
        # Need to figure out where to fit these in...
            # Could use both to replace 4A? or 4B? Or add further subplots to 4?
            # Both side-by-side with shared axes to replace 4A?
        # Using: individual - mse_linear/ae, linear/ae_interpolation_loss_mean
        
    ### Supplementary figures ###
    
    ### Individual vs. transfer reconstruction - full scale ###
    fig, ax = plt.subplots()
    
    mse_combined = np.concatenate((
        individual['mse_linear'],
        individual['mse_ae'],
        transfer['mse_linear'],
        transfer['mse_ae']
    ))
    
    model_type = np.concatenate((
        np.zeros(len(individual['mse_linear'])),
        np.ones(len(individual['mse_ae'])),
        np.zeros(len(transfer['mse_linear'])),
        np.ones(len(transfer['mse_ae']))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(individual['mse_linear'])),
        np.zeros(len(individual['mse_ae'])),
        np.ones(len(transfer['mse_linear'])),
        np.ones(len(transfer['mse_ae']))
    ))
    
    jitter = 0.1
    x_pos = np.concatenate((
        np.zeros(len(individual['mse_linear'])) + np.random.standard_normal(len(individual['mse_linear']))*jitter,
        np.ones(len(individual['mse_ae']))*0.5 + np.random.standard_normal(len(individual['mse_ae']))*jitter,
        np.ones(len(transfer['mse_linear']))*2.5 + np.random.standard_normal(len(individual['mse_linear']))*jitter,
        np.ones(len(transfer['mse_ae']))*3 + np.random.standard_normal(len(individual['mse_ae']))*jitter
    ))
    
    df = pd.DataFrame([mse_combined, model_type, data_source, x_pos])
    df = df.T
    df.columns = ['MSE', 'Model', 'Source', 'X Position']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'SAE', inplace=True)
    df['Source'].replace(0, 'Individual', inplace=True)
    df['Source'].replace(1, 'Transfer', inplace=True)
    
    sns.scatterplot(
        data = df,
        x = 'X Position',
        y = 'MSE',
        hue = 'Model',
        hue_order = ['PCA', 'SAE'],
        style = 'Model',
        markers = ['o', 'X'],
        palette = {'PCA': colour_pca, 'SAE': colour_sae},
        x_jitter = 0.2,
        ax = ax
    )
    
    ax.set(
        ylabel = 'MSE',
        xlabel = '',
        xticks = [0.25, 2.75],
        xticklabels = ['Individual', 'Transfer'],
    )
    
    # Create custom legend
    custom = [
        Line2D([], [], marker='o', color=colour_pca, linestyle='None'),
        Line2D([], [], marker='X', color=colour_sae, linestyle='None'),
    ]
    
    ax.legend(custom, ['PCA', 'SAE'], loc='upper center', ncol=2, bbox_to_anchor=(0,0,1,1.3), frameon=False)
    
    sns.despine(ax=ax, offset=10, trim=True)
    fig.tight_layout()
    saveFig(fig, 'Supplementary_2A')
    
    ### Individual vs. transfer interpolation - full scale ###
    individual_linear = individual['linear_interpolation_loss_mean']
    individual_ae = individual['ae_interpolation_loss_mean']
    transfer_linear = transfer['linear_interpolation_loss_mean']
    transfer_ae = transfer['ae_interpolation_loss_mean']
    
    fig, ax = plt.subplots()
    
    interpolation_combined = np.concatenate((
        individual_linear,
        individual_ae,
        transfer_linear,
        transfer_ae
    ))
    
    model_type = np.concatenate((
        np.zeros(len(individual_linear)),
        np.ones(len(individual_ae)),
        np.zeros(len(transfer_linear)),
        np.ones(len(transfer_ae))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(individual_linear)),
        np.zeros(len(individual_ae)),
        np.ones(len(transfer_linear)),
        np.ones(len(transfer_ae))
    ))
    
    jitter = 0.1
    x_pos = np.concatenate((
        np.zeros(len(individual_linear)) + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(individual_ae))*0.5 + np.random.standard_normal(len(individual_ae))*jitter,
        np.ones(len(transfer_linear))*2.5 + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(transfer_ae))*3 + np.random.standard_normal(len(individual_ae))*jitter
    ))
    
    df = pd.DataFrame([interpolation_combined, model_type, data_source, x_pos])
    df = df.T
    df.columns = ['Interpolation', 'Model', 'Source', 'X Position']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'SAE', inplace=True)
    df['Source'].replace(0, 'Individual', inplace=True)
    df['Source'].replace(1, 'Transfer', inplace=True)
    
    sns.scatterplot(
        data = df,
        x = 'X Position',
        y = 'Interpolation',
        hue = 'Model',
        hue_order = ['PCA', 'SAE'],
        style = 'Model',
        markers = ['o', 'X'],
        palette = {'PCA': colour_pca, 'SAE': colour_sae},
        ax = ax
    )
    
    '''sns.violinplot(
        data = df,
        x = 'Source',
        y = 'Interpolation',
        hue = 'Model',
        ax = ax,
        cut = 0
    )'''
    
    ax.set(
        ylabel = 'MSE',
        xlabel = '',
        xticks = [0.25, 2.75],
        xticklabels = ['Individual', 'Transfer']
    )
    
    # Create custom legend
    custom = [
        Line2D([], [], marker='o', color=colour_pca, linestyle='None'),
        Line2D([], [], marker='X', color=colour_sae, linestyle='None')
    ]
    
    ax.legend(custom, ['PCA', 'SAE'], loc='upper center', ncol=2, bbox_to_anchor=(0,0,1,1.3), frameon=False)
    
    fig.tight_layout()
    sns.despine(ax=ax, offset=10, trim=True)
    saveFig(fig, 'Supplementary_4A')
    
    ### Individual vs. transfer direct & interpolated - full scale ###
    fig, ax = plt.subplots(1,2, sharey='row', figsize=(10,5))
    
    direct_linear = individual['mse_linear']
    direct_ae = individual['mse_ae']
    interpolated_linear = individual['linear_interpolation_loss_mean']
    interpolated_ae = individual['ae_interpolation_loss_mean']
    
    interpolation_combined = np.concatenate((
        direct_linear,
        direct_ae,
        interpolated_linear,
        interpolated_ae
    ))
    
    model_type = np.concatenate((
        np.zeros(len(direct_linear)),
        np.ones(len(direct_ae)),
        np.zeros(len(interpolated_linear)),
        np.ones(len(interpolated_ae))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(direct_linear)),
        np.zeros(len(direct_ae)),
        np.ones(len(interpolated_linear)),
        np.ones(len(interpolated_ae))
    ))
    
    jitter = 0.1
    x_pos = np.concatenate((
        np.zeros(len(individual_linear)) + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(individual_ae))*0.5 + np.random.standard_normal(len(individual_ae))*jitter,
        np.ones(len(transfer_linear))*2.5 + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(transfer_ae))*3 + np.random.standard_normal(len(individual_ae))*jitter
    ))
    
    df = pd.DataFrame([interpolation_combined, model_type, data_source, x_pos])
    df = df.T
    df.columns = ['Interpolation', 'Model', 'Interp', 'X Position']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'SAE', inplace=True)
    df['Interp'].replace(0, 'Direct', inplace=True)
    df['Interp'].replace(1, 'Interpolated', inplace=True)
    
    sns.scatterplot(
        data = df,
        x = 'X Position',
        y = 'Interpolation',
        hue = 'Model',
        hue_order = ['PCA', 'SAE'],
        style = 'Model',
        markers = ['o', 'X'],
        palette = {'PCA': colour_pca, 'SAE': colour_sae},
        ax = ax[0]
    )
    '''sns.violinplot(
        data = df,
        x = 'Interp',
        y = 'Interpolation',
        hue = 'Model',
        ax = ax[0],
        cut = 0
    )'''
    
    ax[0].set(
        title = 'Individual',
        ylabel = 'MSE',
        xlabel = '',
        xticks = [0.25, 2.75],
        xticklabels = ['Direct', 'Interpolated']
    )
    
    # Create custom legend
    custom = [
        Line2D([], [], marker='o', color=colour_pca, linestyle='None'),
    ]
    
    ax[0].legend(custom, ['PCA'], loc='upper right', ncol=2, bbox_to_anchor=(0,0,1,1.3), frameon=False)
    
    direct_linear = transfer['mse_linear']
    direct_ae = transfer['mse_ae']
    interpolated_linear = transfer['linear_interpolation_loss_mean']
    interpolated_ae = transfer['ae_interpolation_loss_mean']
    
    interpolation_combined = np.concatenate((
        direct_linear,
        direct_ae,
        interpolated_linear,
        interpolated_ae
    ))
    
    model_type = np.concatenate((
        np.zeros(len(direct_linear)),
        np.ones(len(direct_ae)),
        np.zeros(len(interpolated_linear)),
        np.ones(len(interpolated_ae))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(direct_linear)),
        np.zeros(len(direct_ae)),
        np.ones(len(interpolated_linear)),
        np.ones(len(interpolated_ae))
    ))
    
    jitter = 0.1
    x_pos = np.concatenate((
        np.zeros(len(individual_linear)) + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(individual_ae))*0.5 + np.random.standard_normal(len(individual_ae))*jitter,
        np.ones(len(transfer_linear))*2.5 + np.random.standard_normal(len(individual_linear))*jitter,
        np.ones(len(transfer_ae))*3 + np.random.standard_normal(len(individual_ae))*jitter
    ))
    
    df = pd.DataFrame([interpolation_combined, model_type, data_source, x_pos])
    df = df.T
    df.columns = ['Interpolation', 'Model', 'Interp', 'X Position']
    
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'SAE', inplace=True)
    df['Interp'].replace(0, 'Direct', inplace=True)
    df['Interp'].replace(1, 'Interpolated', inplace=True)
    
    sns.scatterplot(
        data = df,
        x = 'X Position',
        y = 'Interpolation',
        hue = 'Model',
        hue_order = ['PCA', 'SAE'],
        style = 'Model',
        markers = ['o', 'X'],
        palette = {'PCA': colour_pca, 'SAE': colour_sae},
        ax = ax[1]
    )
    '''sns.violinplot(
        data = df,
        x = 'Interp',
        y = 'Interpolation',
        hue = 'Model',
        ax = ax[1],
        cut = 0
    )'''
    
    ax[1].set(
        title = 'Transfer',
        xlabel = '',
        ylabel = '',
        xticks = [0.25, 2.75],
        xticklabels = ['Direct', 'Interpolated']
    )
    ax[1].get_legend().remove()
    
    # Create custom legend
    custom = [
        Line2D([], [], marker='X', color=colour_sae, linestyle='None')
    ]
    
    ax[1].legend(custom, ['SAE'], loc='upper left', ncol=2, bbox_to_anchor=(0,0,1,1.3), frameon=False)
    
    sns.despine(ax=ax[0], offset=10, trim=True)
    sns.despine(ax=ax[1], offset=10, trim=True)
    ax[1].axes.get_yaxis().set_visible(False)
    ax[1].spines['left'].set_visible(False)
    fig.tight_layout()
    saveFig(fig, 'Supplementary_4B')

    print("Done")