'''
Generate final plots

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
11/11/2021

Loads all data
Creates final figures
'''
# Imports
from resultshandling import *
from plotting import *

# Load processed data
data = load_results('processed_results')
individual = data['individual']
transfer = data['transfer']
grandaverage = data['grandaverage']

# Load supplementary data
supplementary = load_results('supplementary_results')

# Generate plots
generate_plots(individual, transfer, grandaverage, supplementary)