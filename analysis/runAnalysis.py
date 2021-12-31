'''
Run analyses for hand biomechanics data

Oxford Neural Intefacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
01/08/2021

Runs analyses for all participants & saves results
'''
# Import data & routines
from participants import *
from analysis_individual import *
from analysis_group import *
from analysis_grandaverage import *
from analysis_transfer import *

# Set flags for defining behaviour
flags ={
    'load_from_disk': False, # Whether to skip preprocessing & load prepared data
    'save_data_to_disk': True # Whether to save preprocessd data to disk (to allow later loading for quicker analyses)
}

# Start logging output
print('Running hand biomechanics analysis...')

'''
# Run individual analysis for all participants
print('Running individual analyses...')
flags.update({'is_transfer': False})
for p_id, participant in enumerate(participants):
    print(f"Participant {p_id+1}/{len(participants)} {((p_id+1)/len(participants))*100:.2f}%")
    run_analysis_individual(participant, flags)
    
# Run group analysis using individual data
print('Running group analyses...')
run_analysis_group(participants, flags)
'''
# Run grand-average analyses
print('Running grand average analyses...')
flags.update({'is_transfer': False})
run_analysis_grandaverage(participants, flags)

'''
# Run transfer analysis
print('Running transfer analyses...')
flags.update({'is_transfer': True})
run_analysis_transfer(participants, flags)
'''
# Show confirmation of finishing
print('Done')