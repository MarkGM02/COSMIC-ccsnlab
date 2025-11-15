#loop through metallicities. Create initial population. Put it into initial_pops

from cosmic.sample.initialbinarytable import InitialBinaryTable
import pandas as pd
import numpy as np

def main():
    metallicities = [
        0.0001, 0.000122, 0.000148, 0.00018,  0.00022,  0.000267, 0.000325,
        0.000396, 0.000482, 0.000587, 0.000715, 0.00087, 0.001059, 0.00129,
        0.00157,  0.001911, 0.002326, 0.002832, 0.003448, 0.004197, 0.005109,
        0.00622,  0.007572, 0.009218, 0.011221, 0.01366,  0.016629, 0.020243,
        0.024644, 0.03
        ]
    keep_singles = True
    final_kstars = list(range(16))
    np.random.seed(16)
    size = 1E6
    
    for metallicity in metallicities:
        InitialBinaries, mass_singles, mass_binaries, n_singles, n_binaries = InitialBinaryTable.sampler(
            'independent',
            final_kstars,
            final_kstars,
            binfrac_model='offner22',
            primary_model='kroupa01',
            ecc_model='sana12',
            porb_model='sana12',
            qmin=-1,
            m2_min=0.08,
            SF_start=13700.0,
            SF_duration=0.0,
            met=metallicity,
            size=size,
            keep_singles=keep_singles
        )
        path = f'initial_pops_final/pop_met_{metallicity}.h5'
        with pd.HDFStore(path, 'w') as store:
            store['InitialBinaries'] = InitialBinaries
        print("Population for metallicity {metallicity} initialized\n")
    
    print("All populations initialized!\n")

main()