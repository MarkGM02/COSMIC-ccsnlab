import pandas as pd

sn_map = { #maps numeric SN type codes in COSMIC to string labels
    1: 'CCSN',
    2: 'ECSN',
    3: 'Ultra-stripped',
    4: 'AIC',
    5: 'Merger induced collapse',
    6: 'Pulsational pair instability',
    7: 'Pair instability',
    0: None
}

def sn_types(input, II_h_thresh = 0.033):
    """Label SNe in the dataframe as either Type I, Type II, or some exotic type based on the SN type column.

    Args:
        input (pd.DataFrame): DataFrame containing SN data. Must include at minimum the columns 'SN_1', 'SN_2', 'sn_1_m_h_ejecta', and 'sn_2_m_h_ejecta'.
        II_h_thresh (float): Threshold for separating Type I from Type II SNe based on H ejecta mass.

    Returns:
        pd.DataFrame: DataFrame with a new columns 'sn_1_type' and 'sn_2_type' containing 'I', 'II', or exotic type labels.
    """

    data = input.copy()
    for sn in [1,2]:
        #initialize the type column
        sn_type_string = f'sn_{sn}_type'
        data[sn_type_string] = pd.Series(index=input.index, dtype='object')

        cc_mask = input[f'SN_{sn}'].isin([1, 3])
        type_ii_mask = cc_mask & (input[f'sn_{sn}_m_h_ejecta'].fillna(0) > II_h_thresh)
        type_i_mask  = cc_mask & ~type_ii_mask
        
        # write back the types
        data.loc[type_i_mask, sn_type_string] = 'I'
        data.loc[type_ii_mask, sn_type_string] = 'II'
        # exotic paths: copy from lookup table
        exotic_mask = ~cc_mask
        data.loc[exotic_mask, sn_type_string] = input.loc[exotic_mask, f'SN_{sn}'].map(sn_map)

    return data


def sn_subtypes(input,
                II_h_thresh = 0.033,
                IIb_IIL_h_thresh = 0.91,
                IIL_IIP_h_thresh = 4.5,
                Ib_Ic_he_thresh = 0.14,
                m_loss_thresh = -1e-4,
                Ic_scheme = 'relative', Ic_ratio=0.43,
                IIP_scheme='branch IIP first'
                ):

    """Classify SNe from the input dataframe. Requires the columns 'sn_1_type', 'sn_2_type', 'sn_1_m_h_ejecta', 'sn_2_m_h_ejecta',
       'sn_1_m_he_ejecta', 'sn_2_m_he_ejecta', 'sn_1_m_ejecta', 'sn_2_m_ejecta', 'sn_1_max_loss_rate', 'sn_2_max_loss_rate', 'sn_1_donor_kstars',
       and 'sn_2_donor_kstars' for all functionality.

    Args:
        input (pd.DataFrame): DataFrame containing by system ejecta and mass loss data.
        II_h_thresh (float): Threshold for separating Type I from Type II SNe based on H ejecta mass.
        IIb_IIL_h_thresh (float): Threshold for separating Type IIL from Type IIb SNe based on H ejecta mass.
        IIL_IIP_h_thresh (float): Threshold for separating Type IIL from Type IIP SNe based on H ejecta mass.
        Ic_scheme (str): Scheme for classifying Type Ic SNe, either 'absolute', 'relative', or 'mass transfer'.
        Ib_Ic_he_thresh (float): Threshold for separating Type Ib from Type Ic SNe based on He mass. Used only if Ic_scheme is 'absolute'.
        Ic_ratio (float): Ratio of He mass to total ejecta mass for Type Ic SNe, used if Ic_scheme is 'relative'.
        IIP_scheme (str): Scheme for classifying Type IIP SNe, either 'branch IIP first' or 'branch IIn first'.
        m_loss_thresh (float): Threshold for separating Type IIn SNe based on mass loss rate.
        
        Returns:
        pd.DataFrame: DataFrame with new columns 'sn_1_subtype' and 'sn_2_subtype' containing the SN subtype labels.
    """

    data = input.copy()

    for sn in [1,2]:

        #initialize the subtype column
        sn_subtype_string = f'sn_{sn}_subtype'
        data[sn_subtype_string] = pd.Series(index=input.index, dtype='object')

        # get the ejecta masses, and mass loss
        m_h_ejecta = data[f'sn_{sn}_m_h_ejecta']
        m_he_ejecta = data[f'sn_{sn}_m_he_ejecta']
        m_ejecta = data[f'sn_{sn}_m_ejecta']
        m_loss_rate = data[f'sn_{sn}_max_loss_rate']

        #get a mask for type Is and type IIs
        type_i_mask = m_h_ejecta < II_h_thresh
        type_ii_mask = m_h_ejecta >= II_h_thresh

        #get a mask for mass loss -- note that in COSMIC, mass loss is negative
        type_n_mask = m_loss_rate <= m_loss_thresh

        # Type I subtypes
        if Ic_scheme == 'absolute':
            Ic_mask = type_i_mask & (m_he_ejecta < Ib_Ic_he_thresh)
        elif Ic_scheme == 'relative':
            Ic_mask = type_i_mask & (m_he_ejecta < Ic_ratio * m_ejecta)
        elif Ic_scheme == 'mass transfer':
            # in this scheme, we assume that if the star donated mass as a stripped star (kstar=7,8,9) it is a Ic
            Ic_mask = data[f'sn_{sn}_donor_kstars'].astype(str).str.split('-').apply(lambda lst: any(x in ['7', '8', '9'] for x in lst))
        else:
            raise ValueError("Ic_scheme must be either 'absolute', 'relative', or 'mass transfer'")
        Ib_mask = type_i_mask & (~Ic_mask)

        # Type II subtypes
        if IIP_scheme == 'branch IIn first':
            IIP_mask = type_ii_mask & (m_h_ejecta >= IIL_IIP_h_thresh) & (~type_n_mask)
        elif IIP_scheme == 'branch IIP first':
            IIP_mask = type_ii_mask & (m_h_ejecta >= IIL_IIP_h_thresh)
        else:
            raise ValueError("IIP_scheme must be either 'branch IIP first' or 'branch IIn first'")

        IIn_mask = type_ii_mask & type_n_mask & (~IIP_mask)
        IIb_mask = type_ii_mask & (m_h_ejecta < IIb_IIL_h_thresh) & (~IIn_mask)
        IIL_mask = type_ii_mask & (~IIP_mask) & (~IIb_mask) & (~IIn_mask)

        # write back the subtypes
        data.loc[Ic_mask, sn_subtype_string] =  'Ic'
        data.loc[Ib_mask, sn_subtype_string] =  'Ib'
        data.loc[IIn_mask, sn_subtype_string] = 'IIn'
        data.loc[IIb_mask, sn_subtype_string] = 'IIb'
        data.loc[IIL_mask, sn_subtype_string] = 'IIL'
        data.loc[IIP_mask, sn_subtype_string] = 'IIP'

        # non type I and II should be whatever exotic type is in sn_type
        sn_type = data[f'sn_{sn}_type']
        non_I_II_mask = ~sn_type.isin(['I', 'II'])
        data.loc[non_I_II_mask, sn_subtype_string] = sn_type

    return data
