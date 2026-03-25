"""Module for finding out how much of the mass was lost to neutrinos so we may calculate ejecta as M - M_rem - M_neutrino."""

import numpy as np

def neutrino_mass_loss(remnant_mass, rembar_massloss=0.5):
    """
    Apply COSMIC's neutrino mass loss prescription, where the mass loss is
    limited to rembar_massloss. If this is positive, it is an absolute mass limit.
    If negative, it is a fraction of the remnant mass.
    
    :param remnant_mass: The mass of the remnant before neutrino mass loss is applied.
    :param rembar_massloss: The maximum mass loss due to neutrinos. If positive, this is an absolute mass limit. If negative, this is a fraction of the remnant mass.
    :return: The mass of the remnant after neutrino mass loss is applied.
    """
    reduced_remnant_mass = 6.6666667*(np.sqrt(1.0 + 0.3* remnant_mass) - 1.0)
    mass_diff = remnant_mass - reduced_remnant_mass

    if rembar_massloss >= 0:
        return remnant_mass - rembar_massloss if mass_diff > rembar_massloss else reduced_remnant_mass
    else:
        return remnant_mass * (1 + rembar_massloss) if mass_diff > -rembar_massloss * remnant_mass else reduced_remnant_mass

def invert_neutrino_mass_loss(final_remnant_mass, rembar_massloss=0.5):
    """
    Invert the neutrino mass loss prescription to find the original remnant mass before neutrino mass loss was applied.
    
    :param final_remnant_mass: The mass of the remnant after neutrino mass loss is applied.
    :param rembar_massloss: The maximum mass loss due to neutrinos. If positive, this is an absolute mass limit. If negative, this is a fraction of the remnant mass.
    :return: The original remnant mass before neutrino mass loss was applied.
    """
    
    if rembar_massloss == 0: return final_remnant_mass

    M_i = (1 / 0.3) * ((final_remnant_mass / 6.6666667 + 1)**2 - 1)
    mass_diff = M_i - final_remnant_mass

    if rembar_massloss >= 0:
        if mass_diff > rembar_massloss:
            return final_remnant_mass + rembar_massloss
        else:
            return M_i
    else:
        if mass_diff > -rembar_massloss * M_i:
            return final_remnant_mass / (1 + rembar_massloss)
        else:
            return M_i

def get_neutrino_mass_loss(final_remnant_mass, rembar_massloss=0.5):
    """
    Calculate the mass lost to neutrinos based on the final remnant mass after neutrino mass loss is applied.
    
    :param final_remnant_mass: The mass of the remnant after neutrino mass loss is applied.
    :param rembar_massloss: The maximum mass loss due to neutrinos. If positive, this is an absolute mass limit. If negative, this is a fraction of the remnant mass.
    :return: The mass lost to neutrinos.
    """
    original_remnant_mass = invert_neutrino_mass_loss(final_remnant_mass, rembar_massloss=rembar_massloss)
    return original_remnant_mass - final_remnant_mass