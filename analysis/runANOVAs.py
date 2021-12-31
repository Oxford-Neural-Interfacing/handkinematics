'''
Run ANOVAs for biomechanics data

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
10/11/2021

Creates tables & runs ANOVAs using group-level data
Prints ANOVA tables
'''
# Imports
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel as ttest
#from scipy.stats import wilcoxon as ttest

def run_ANOVA(results_individual, results_transfer):
    '''
    Runs ANOVA tests & prints results
    
    Args:
        results_individual: dict of individual-level results
        results_transfer: dict of transfer results
    '''
    ### Reconstruction: model type, data source ###
    # Combine data
    mse_combined = np.concatenate((
        results_individual['mse_linear'],
        results_individual['mse_ae'],
        results_transfer['mse_linear'],
        results_transfer['mse_ae']
    ))
    
    model_type = np.concatenate((
        np.zeros(len(results_individual['mse_linear'])),
        np.ones(len(results_individual['mse_ae'])),
        np.zeros(len(results_transfer['mse_linear'])),
        np.ones(len(results_transfer['mse_ae']))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(results_individual['mse_linear'])),
        np.zeros(len(results_individual['mse_ae'])),
        np.ones(len(results_transfer['mse_linear'])),
        np.ones(len(results_transfer['mse_ae']))
    ))
    
    IDs = np.linspace(1, len(results_individual['mse_linear']), num=len(results_individual['mse_linear']))
    IDs_all = np.concatenate((
        IDs,
        IDs,
        IDs,
        IDs
    ))
    
    # Create dataframe
    df = pd.DataFrame([
        IDs_all,
        mse_combined,
        model_type,
        data_source
    ])
    df = df.T
    df.columns = ['ID', 'MSE', 'Model', 'Source']
    
    # Replace labels
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'RNN-AE', inplace=True)
    df['Source'].replace(0, 'Individual', inplace=True)
    df['Source'].replace(1, 'Transfer', inplace=True)

    # Run ANOVA
    anova = AnovaRM(df, 'MSE', 'ID', within=['Model', 'Source'])
    result = anova.fit()
    
    # Output result
    print("----- ANOVA: Reconstruction -----")
    print(result)
    
    print ("----- Post-hoc: Reconstruction -----")
    # PCA vs AE
    pca = np.concatenate((results_individual['mse_linear'], results_transfer['mse_linear']))
    ae = np.concatenate((results_individual['mse_ae'], results_transfer['mse_ae']))
    print(f"PCA vs. AE: {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Individual vs. transfer
    individual = np.concatenate((results_individual['mse_linear'], results_individual['mse_ae']))
    transfer = np.concatenate((results_transfer['mse_linear'], results_transfer['mse_ae']))
    print(f"Individual vs. transfer: {individual.mean()} +/- {individual.std()} vs. {transfer.mean()} +/- {transfer.std()}, p = {ttest(individual,transfer)[1]}\n")
    
    # PCA vs. AE, individual
    pca = np.array(results_individual['mse_linear'])
    ae = np.array(results_individual['mse_ae'])
    print(f"PCA vs. AE (individual): {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # PCA vs. AE, transfer
    pca = np.array(results_transfer['mse_linear'])
    ae = np.array(results_transfer['mse_ae'])
    print(f"PCA vs. AE (transfer): {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Individual vs. transfer, PCA
    individual = np.array(results_individual['mse_linear'])
    transfer = np.array(results_transfer['mse_linear'])
    print(f"Individual vs. transfer (PCA): {individual.mean()} +/- {individual.std()} vs. {transfer.mean()} +/- {transfer.std()}, p = {ttest(individual,transfer)[1]}\n")
    
    # Individual vs. transfer, AE
    individual = np.array(results_individual['mse_ae'])
    transfer = np.array(results_transfer['mse_ae'])
    print(f"Individual vs. transfer (AE): {individual.mean()} +/- {individual.std()} vs. {transfer.mean()} +/- {transfer.std()}, p = {ttest(individual,transfer)[1]}\n")
    
    # (transfer - individual), PCA vs. AE
    pca = np.array(results_transfer['mse_linear']) - np.array(results_individual['mse_linear'])
    ae = np.array(results_transfer['mse_ae']) - np.array(results_individual['mse_ae'])
    print(f"PCA vs. AE (deterioration): {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Individual PCA vs. transfer AE
    pca = np.array(results_individual['mse_linear'])
    ae = np.array(results_transfer['mse_ae'])
    print(f"Individual PCA vs. transfer AE: {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    ### Classification: model type, data source ###
    # Combine data
    accuracy_combined = np.concatenate((
        [r[0] for r in results_individual['accuracy_linear_max']],
        [r[0] for r in results_individual['accuracy_ae_max']],
        [r[0] for r in results_transfer['accuracy_linear_max']],
        [r[0] for r in results_transfer['accuracy_ae_max']]
    ))
    
    model_type = np.concatenate((
        np.zeros(len(results_individual['accuracy_linear_max'])),
        np.ones(len(results_individual['accuracy_ae_max'])),
        np.zeros(len(results_transfer['accuracy_linear_max'])),
        np.ones(len(results_transfer['accuracy_ae_max']))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(results_individual['accuracy_linear_max'])),
        np.zeros(len(results_individual['accuracy_ae_max'])),
        np.ones(len(results_transfer['accuracy_linear_max'])),
        np.ones(len(results_transfer['accuracy_ae_max']))
    ))
    
    IDs = np.linspace(1, len(results_individual['accuracy_linear_max']), num=len(results_individual['accuracy_linear_max']))
    IDs_all = np.concatenate((
        IDs,
        IDs,
        IDs,
        IDs
    ))
    
    # Create dataframe
    df = pd.DataFrame([
        IDs_all,
        accuracy_combined,
        model_type,
        data_source
    ])
    df = df.T
    df.columns = ['ID', 'Accuracy', 'Model', 'Source']
    
    # Replace labels
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'RNN-AE', inplace=True)
    df['Source'].replace(0, 'Individual', inplace=True)
    df['Source'].replace(1, 'Transfer', inplace=True)
    
    # Run ANOVA
    anova = AnovaRM(df, 'Accuracy', 'ID', within=['Model', 'Source'])
    result = anova.fit()
    
    # Output result
    print("----- ANOVA: Classification -----")
    print(result)
    
    print("----- Post-hoc: Classification -----")
    # PCA vs. AE
    pca = np.concatenate((
        [r[0] for r in results_individual['accuracy_linear_max']],
        [r[0] for r in results_transfer['accuracy_linear_max']]
    ))
    ae = np.concatenate((
        [r[0] for r in results_individual['accuracy_ae_max']],
        [r[0] for r in results_transfer['accuracy_ae_max']]
    ))
    print(f"PCA vs. AE: {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Individual vs. transfer
    individual = np.concatenate((
        [r[0] for r in results_individual['accuracy_linear_max']],
        [r[0] for r in results_individual['accuracy_ae_max']]
    ))
    transfer = np.concatenate((
        [r[0] for r in results_transfer['accuracy_linear_max']],
        [r[0] for r in results_transfer['accuracy_ae_max']]
    ))
    print(f"Individual vs. transfer: {individual.mean()} +/- {individual.std()} vs. {transfer.mean()} +/- {transfer.std()}, p = {ttest(individual,transfer)[1]}\n")
    
    # PCA vs. AE (individual)
    pca = np.array([r[0] for r in results_individual['accuracy_linear_max']])
    ae = np.array([r[0] for r in results_individual['accuracy_ae_max']])
    print(f"PCA vs. AE (individual): {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # PCA vs. AE (transfer)
    pca = np.array([r[0] for r in results_transfer['accuracy_linear_max']])
    ae = np.array([r[0] for r in results_transfer['accuracy_ae_max']])
    print(f"PCA vs. AE (transfer): {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Individual vs. transfer (PCA)
    individual = np.array([r[0] for r in results_individual['accuracy_linear_max']])
    transfer = np.array([r[0] for r in results_transfer['accuracy_linear_max']])
    print(f"Individual vs. transfer (PCA): {individual.mean()} +/- {individual.std()} vs. {transfer.mean()} +/- {transfer.std()}, p = {ttest(individual,transfer)[1]}\n")
    
    # Individual vs. transfer (AE)
    individual = np.array([r[0] for r in results_individual['accuracy_ae_max']])
    transfer = np.array([r[0] for r in results_transfer['accuracy_ae_max']])
    print(f"Individual vs. transfer (AE): {individual.mean()} +/- {individual.std()} vs. {transfer.mean()} +/- {transfer.std()}, p = {ttest(individual,transfer)[1]}\n")
    
    # (individual - transfer) PCA vs. AE
    pca = np.array([r[0] for r in results_individual['accuracy_linear_max']]) - np.array([r[0] for r in results_transfer['accuracy_linear_max']])
    ae = np.array([r[0] for r in results_individual['accuracy_ae_max']]) - np.array([r[0] for r in results_transfer['accuracy_ae_max']])
    print(f"PCA vs. AE (deterioration): {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Individual PCA vs. transfer AE
    pca = np.array([r[0] for r in results_individual['accuracy_linear_max']])
    ae = np.array([r[0] for r in results_transfer['accuracy_ae_max']])
    print(f"Individual PCA vs. transfer AE: {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    ### Interpolation: model type, data source ###
    # Combine data
    interpolation_combined = np.concatenate((
        results_individual['linear_interpolation_loss_mean'],
        results_individual['ae_interpolation_loss_mean'],
        results_transfer['linear_interpolation_loss_mean'],
        results_transfer['ae_interpolation_loss_mean']
    ))
    
    model_type = np.concatenate((
        np.zeros(len(results_individual['linear_interpolation_loss_mean'])),
        np.ones(len(results_individual['ae_interpolation_loss_mean'])),
        np.zeros(len(results_transfer['linear_interpolation_loss_mean'])),
        np.ones(len(results_transfer['ae_interpolation_loss_mean']))
    ))
    
    data_source = np.concatenate((
        np.zeros(len(results_individual['linear_interpolation_loss_mean'])),
        np.zeros(len(results_individual['ae_interpolation_loss_mean'])),
        np.ones(len(results_transfer['linear_interpolation_loss_mean'])),
        np.ones(len(results_transfer['ae_interpolation_loss_mean']))
    ))
    
    IDs = np.linspace(1, len(results_individual['linear_interpolation_loss_mean']), num=len(results_individual['linear_interpolation_loss_mean']))
    IDs_all = np.concatenate((
        IDs,
        IDs,
        IDs,
        IDs
    ))
    
    # Create dataframe
    df = pd.DataFrame([
        IDs_all,
        interpolation_combined,
        model_type,
        data_source
    ])
    df = df.T
    df.columns = ['ID', 'Interpolation', 'Model', 'Source']
    
    # Replace labels
    df['Model'].replace(0, 'PCA', inplace=True)
    df['Model'].replace(1, 'RNN-AE', inplace=True)
    df['Source'].replace(0, 'Individual', inplace=True)
    df['Source'].replace(1, 'Transfer', inplace=True)

    # Run ANOVA
    anova = AnovaRM(df, 'Interpolation', 'ID', within=['Model', 'Source'])
    result = anova.fit()
    
    # Output result
    print("----- ANOVA: Interpolation -----")
    print(result)
    
    print("----- Post-hoc: Interpolation -----")
    # PCA vs. AE
    pca = np.concatenate((results_individual['linear_interpolation_loss_mean'], results_transfer['linear_interpolation_loss_mean']))
    ae = np.concatenate((results_individual['ae_interpolation_loss_mean'], results_transfer['ae_interpolation_loss_mean']))
    print(f"PCA vs. AE: {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Individual vs. transfer
    individual = np.concatenate((results_individual['linear_interpolation_loss_mean'], results_individual['ae_interpolation_loss_mean']))
    transfer = np.concatenate((results_transfer['linear_interpolation_loss_mean'], results_transfer['ae_interpolation_loss_mean']))
    print(f"Individual vs. transfer: {individual.mean()} +/- {individual.std()} vs. {transfer.mean()} +/- {transfer.std()}, p = {ttest(individual,transfer)[1]}\n")
    
    # PCA vs. AE, individual
    pca = np.array(results_individual['linear_interpolation_loss_mean'])
    ae = np.array(results_individual['ae_interpolation_loss_mean'])
    print(f"PCA vs. AE (individual): {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # PCA vs. AE, transfer
    pca = np.array(results_transfer['linear_interpolation_loss_mean'])
    ae = np.array(results_transfer['ae_interpolation_loss_mean'])
    print(f"PCA vs. AE (transfer): {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Individual vs. transfer, PCA
    individual = np.array(results_individual['linear_interpolation_loss_mean'])
    transfer = np.array(results_transfer['linear_interpolation_loss_mean'])
    print(f"Individual vs. transfer (PCA): {individual.mean()} +/- {individual.std()} vs. {transfer.mean()} +/- {transfer.std()}, p = {ttest(individual,transfer)[1]}\n")
    
    # Individual vs. transfer, AE
    individual = np.array(results_individual['ae_interpolation_loss_mean'])
    transfer = np.array(results_transfer['ae_interpolation_loss_mean'])
    print(f"Individual vs. transfer (AE): {individual.mean()} +/- {individual.std()} vs. {transfer.mean()} +/- {transfer.std()}, p = {ttest(individual,transfer)[1]}\n")
    
    # (individual - transfer) PCA vs. AE
    pca = np.array(results_transfer['linear_interpolation_loss_mean']) - np.array(results_individual['linear_interpolation_loss_mean'])
    ae = np.array(results_transfer['ae_interpolation_loss_mean']) - np.array(results_individual['ae_interpolation_loss_mean'])
    print(f"PCA vs. AE (deterioration): {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Individual PCA vs. transfer AE
    pca = np.array(results_individual['linear_interpolation_loss_mean'])
    ae = np.array(results_transfer['ae_interpolation_loss_mean'])
    print(f"Individual PCA vs. transfer AE: {pca.mean()} +/- {pca.std()} vs. {ae.mean()} +/- {ae.std()}; p = {ttest(pca, ae)[1]}\n")
    
    # Direct vs. interpolated, PCA, individual
    direct = np.array(results_individual['mse_linear'])
    interpolated = np.array(results_individual['linear_interpolation_loss_mean'])
    print(f"Direct vs. interpolated (PCA, individual): {direct.mean()} +/- {direct.std()} vs. {interpolated.mean()} +/- {interpolated.std()}; p = {ttest(direct, interpolated)[1]}\n")
    
    # Direct vs. interpolated, AE, individual
    direct = np.array(results_individual['mse_ae'])
    interpolated = np.array(results_individual['ae_interpolation_loss_mean'])
    print(f"Direct vs. interpolated (AE, individual): {direct.mean()} +/- {direct.std()} vs. {interpolated.mean()} +/- {interpolated.std()}; p = {ttest(direct, interpolated)[1]}\n")
    
    # Direct vs. interpolated, PCA, transfer
    direct = np.array(results_transfer['mse_linear'])
    interpolated = np.array(results_transfer['linear_interpolation_loss_mean'])
    print(f"Direct vs. interpolated (PCA, transfer): {direct.mean()} +/- {direct.std()} vs. {interpolated.mean()} +/- {interpolated.std()}; p = {ttest(direct, interpolated)[1]}\n")
    
    # Direct vs. interpolated, AE, transfer
    direct = np.array(results_transfer['mse_ae'])
    interpolated = np.array(results_transfer['ae_interpolation_loss_mean'])
    print(f"Direct vs. interpolated (AE, transfer): {direct.mean()} +/- {direct.std()} vs. {interpolated.mean()} +/- {interpolated.std()}; p = {ttest(direct, interpolated)[1]}\n")
    
    # Direct PCA vs. interpolated AE, individual
    direct = np.array(results_individual['mse_linear'])
    interpolated = np.array(results_individual['ae_interpolation_loss_mean'])
    print(f"Direct PCA vs. interpolated AE (individual): {direct.mean()} +/- {direct.std()} vs. {interpolated.mean()} +/- {interpolated.std()}; p = {ttest(direct, interpolated)[1]}\n")
    
    # Direct PCA vs. interpolated AE, transfer
    direct = np.array(results_transfer['mse_linear'])
    interpolated = np.array(results_transfer['ae_interpolation_loss_mean'])
    print(f"Direct PCA vs. interpolated AE (transfer): {direct.mean()} +/- {direct.std()} vs. {interpolated.mean()} +/- {interpolated.std()}; p = {ttest(direct, interpolated)[1]}\n")
    
    ### Others ###
    
    # Reconstruction: interpolated, model type (?data source)
        # Or do for individual & transfer