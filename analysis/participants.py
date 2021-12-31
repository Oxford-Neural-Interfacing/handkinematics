'''
Participant data structures for hand biomechanics analysis

Oxford Neural Interfacing
Written by Conor Keogh
conor.keogh@nds.ox.ac.uk
01/08/2021

Defines each participant as a datastructure with associated data file
Also defines list of all participants
Can then be imported & used to analyse all participants
'''

### Define participants ###
ck = {
    'subjID': 'ck',
    'f_id': '../data/ck_data.csv'
}

gm = {
    'subjID': 'gm',
    'f_id': '../data/gm_data.csv'
}

lc = {
    'subjID': 'lc',
    'f_id': '../data/lc_data.csv'
}

sc = {
    'subjID': 'sc',
    'f_id': '../data/sc_data.csv'
}

chp = {
    'subjID': 'chp',
    'f_id': '../data/chp_data.csv'
}

cp = {
    'subjID': 'cp',
    'f_id': '../data/cp_data.csv'
}

je = {
    'subjID': 'je',
    'f_id': '../data/je_data.csv'
}

jof = {
    'subjID': 'jof',
    'f_id': '../data/jof_data.csv'
}

nl = {
    'subjID': 'nl',
    'f_id': '../data/nl_data.csv'
}

sm = {
    'subjID': 'sm',
    'f_id': '../data/sm_data.csv'
}

ad = {
    'subjID': 'ad',
    'f_id': '../data/ad_data.csv'
}

az = {
    'subjID': 'az',
    'f_id': '../data/az_data.csv'
}

jf = {
    'subjID': 'jf',
    'f_id': '../data/jf_data.csv'
}

lic = {
    'subjID': 'lic',
    'f_id': '../data/lic_data.csv'
}

ra = {
    'subjID': 'ra',
    'f_id': '../data/ra_data.csv'
}

### Define list of all participants to include ###
participants = [
    ck,
    gm,
    lc,
    sc,
    chp,
    cp,
    je,
    jof,
    nl,
    sm,
    ad,
    az,
    jf,
    lic,
    ra
]
