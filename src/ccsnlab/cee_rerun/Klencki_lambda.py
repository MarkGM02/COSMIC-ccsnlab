import numpy as np
import pandas as pd


def get_nearest_value(param_name, param, data):
    
    values = data[param_name].values
    
    if param_name == '#M/Msun':
        closest_index = (data[param_name] - param).abs().idxmin()        
    else:
        closest_index = (np.log10(data[param_name]) - np.log10(param)).abs().idxmin()
            
    return data.loc[closest_index, param_name]

def get_coeffs(r, M, Z, data):
    #get the closest mass
    closest_mass = get_nearest_value('#M/Msun', M, data)
    #Get the closest Z
    closest_Z = get_nearest_value('Z/Zsun', Z, data)

    row = data[(data['#M/Msun'] == closest_mass) & (data['Z/Zsun'] == closest_Z)]
    if row.empty:
        #we probably have too massive a star -- take the max mass available at this metallicity
        closest_mass = data[data['Z/Zsun'] == closest_Z]['#M/Msun'].max()
        row = data[(data['#M/Msun'] == closest_mass) & (data['Z/Zsun'] == closest_Z)]
        if row.empty: raise ValueError(f"Error for M={M} and Z={Z}.")
    
    row = row.iloc[0]

    if r < row['R12/Rsun']:
        return row['a1'], row['b1'], row['c1'], row['d1']
    elif r < row['R23/Rsun']:
        return row['a2'], row['b2'], row['c2'], row['d2']
    else:
        return row['a3'], row['b3'], row['c3'], row['d3']
        
def get_lambda(r, M, Z):
    data = pd.read_csv("lambda_R_fit.dat", sep='\s+', header='infer', skiprows=1)
    a, b, c, d = get_coeffs(r, M, Z, data)
    log_lambda = a*np.log10(r)**3 + b*np.log10(r)**2 + c*np.log10(r) + d
    return 10**log_lambda if log_lambda < 10 else 10**10