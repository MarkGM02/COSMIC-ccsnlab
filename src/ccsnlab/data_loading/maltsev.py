
import pandas as pd
import numpy as np

def get_masses(metallicity, case, maltsev_mode=0):
    """
    Calculate the masses M1, M2, and M3 based on the Maltsev et al. (2025) prescription for assigning succesful explosions
    and remnant types. The masses are calculated based on the metallicity, the case (A, B, C, S, No_MT, Case_A, Case_B, Case_C),
    and the maltsev_mode (0, 1, or 2).
    
    :param metallicity: The DIMENSIONLESS metallicity of the star (Z/Zsun).
    :param case: The case to use for the calculation (A, B, C, S, No_MT, Case_A, Case_B, Case_C).
    :param maltsev_mode: The extrapolation mode used, see the cosmic docs (0, 1, or 2).
    :return: A tuple containing the masses M1, M2, and M3.
    """

    bounds = pd.DataFrame({ 'M':      ['M1', 'M2', 'M3', 'M1', 'M2', 'M3'],
                            'Z/Zsun': [1,    1,    1,    0.1,  0.1,  0.1],
                            'Case_A': [7.4,  8.4,  15.4, 7.0,  7.4,  13.7],
                            'Case_B': [7.7,  8.3,  15.2, 6.9,  7.9,  13.7],
                            'Case_C': [6.6,  7.1,  13.2, 6.3,  7.1,  12.3],
                            'No_MT':  [6.6,  7.2,  13.0, 6.1,  6.6,  12.9]
    })

    case_map = {
        'A': 'Case_A',
        'B': 'Case_B',
        'C': 'Case_C',
        'S': 'No_MT',
        'No_MT': 'No_MT',
        'Case_A': 'Case_A',
        'Case_B': 'Case_B',
        'Case_C': 'Case_C',
    }
    key = case_map.get(case, None)
    if key is None:
        raise ValueError("case must be one of: A, B, C, S, No_MT, Case_A, Case_B, Case_C")

    if metallicity <= 0:
        raise ValueError("metallicity must be positive (Z/Zsun)")

    log10Z = np.log10(metallicity)
    if maltsev_mode == 0:
        log10Z_bounded = log10Z
    elif maltsev_mode == 1:
        log10Z_bounded = min(max(log10Z, -1.69897), 0.0)
    elif maltsev_mode == 2:
        log10Z_bounded = min(max(log10Z, -1.0), 0.0)
    else:
        raise ValueError("maltsev_mode must be 0, 1, or 2")

    rows_z1 = bounds[bounds['Z/Zsun'] == 1].set_index('M')
    rows_z01 = bounds[bounds['Z/Zsun'] == 0.1].set_index('M')

    def compute_mass(m_label):
        m1 = rows_z1.loc[m_label, key]
        m01 = rows_z01.loc[m_label, key]
        return m1 + (m1 - m01) * log10Z_bounded

    m1 = compute_mass('M1')
    m2 = compute_mass('M2')
    m3 = compute_mass('M3')
    return (m1, m2, m3)
