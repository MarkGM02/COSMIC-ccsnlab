#Solving the delayed fryer remnant mass prescription for the original CO core mass

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def get_proto_core_mass(core_mass):
    """
    Helper function to calculate the proto-core mass based on the original CO core mass in the delayed fryer prescription.
    
    :param core_mass: The original CO core mass.
    :return: The proto-core mass.
    """
    if core_mass <= 3.5:
        return 1.2
    elif core_mass <= 6.0:
        return 1.3
    elif core_mass <= 11.0:
        return 1.4
    else:
        return 1.6

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

def get_remnant_mass(core_mass, total_mass, rembar_massloss=0.5):
    """
    Calculate the remnant mass based on the delayed fryer prescription.
    
    :param core_mass: The original CO core mass.
    :param total_mass: The total mass of the star.
    :param rembar_massloss: The maximum mass loss due to neutrinos. If positive, this is an absolute mass limit. If negative, this is a fraction of the remnant mass.
    :return: A tuple containing the final remnant mass and the mass lost to neutrinos.
    """
    final_mass = None
    proto_core_mass = get_proto_core_mass(core_mass)
    if core_mass < 2.5:
        remnant_mass = proto_core_mass + 0.2
    elif core_mass < 3.5:
        remnant_mass = proto_core_mass + 0.5 * core_mass - 1.05
    elif core_mass < 11.0:
        avar = 0.133 - (0.093 / (total_mass - proto_core_mass))
        bvar = 1.0 - 11.0 * avar
        fallback = avar * core_mass + bvar
        remnant_mass = proto_core_mass + fallback*(total_mass - proto_core_mass)
    else:
        remnant_mass = total_mass

    remnant_mass_final = neutrino_mass_loss(final_mass, rembar_massloss=rembar_massloss)
    mass_lost_to_neutrinos = remnant_mass - remnant_mass_final

    return remnant_mass_final, mass_lost_to_neutrinos