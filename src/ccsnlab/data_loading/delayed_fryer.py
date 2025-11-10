#Solving the delayed fryer remnant mass prescription for the original CO core mass

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def get_proto_core_mass(core_mass):
    if core_mass <= 3.5:
        return 1.2
    elif core_mass <= 6.0:
        return 1.3
    elif core_mass <= 11.0:
        return 1.4
    else:
        return 1.6
    
def get_remnant_mass(core_mass, total_mass):
    #ignore the electron capture case
    rembar_massloss = 0.5

    final_mass = None
    proto_core_mass = get_proto_core_mass(core_mass)
    if core_mass < 2.5:
        final_mass = proto_core_mass + 0.2
    elif core_mass < 3.5:
        final_mass = proto_core_mass + 0.5 * core_mass - 1.05
    elif core_mass < 11.0:
        avar = 0.133 - (0.093 / (total_mass - proto_core_mass))
        bvar = 1.0 - 11.0 * avar
        fallback = avar * core_mass + bvar
        final_mass = proto_core_mass + fallback*(total_mass - proto_core_mass)
    else:
        final_mass = total_mass

    remnant_mass = 6.6666667*(np.sqrt(1.0 + 0.3* final_mass) - 1.0)

    if (final_mass - remnant_mass) >= rembar_massloss:
        remnant_mass = final_mass - rembar_massloss
    
    return remnant_mass

#so the idea is that we can go back to the CO core mass! Only with the total mass, and the remnant mass!
def test_get_remnant_mass(test_masses):
    for core_mass, total_mass in test_masses:
        print(f"Remnant mass for core mass {core_mass} and total mass {total_mass}:", get_remnant_mass(core_mass, total_mass))


def plot(combined_core_mass, total_mass, remnant_mass, CO_core_mass=None):
    core_masses = np.linspace(0, combined_core_mass, num=100)
    remnant_masses = [get_remnant_mass(core_mass, total_mass) for core_mass in core_masses]
    plt.plot(core_masses, remnant_masses, label='Remnant Mass')
    plt.axhline(y=remnant_mass, color='r', linestyle='--', label='Given Remnant Mass')
    plt.axvline(x=combined_core_mass, color='g', linestyle='--', label='Combined Core Mass')
    if CO_core_mass != None: plt.axvline(x=CO_core_mass, color='thistle', linestyle='-', label='CO Core Mass')
    plt.xlabel('Core Mass')
    plt.ylabel('Remnant Mass')
    plt.legend()
    plt.show()

#write a numerical solver

def solve_and_plot(combined_core_mass, total_mass, remnant_mass):
    CO_core_mass = find_CO_core_mass(remnant_mass, total_mass, combined_core_mass)
    if CO_core_mass < 0:
        print("No solution found for the given remnant mass and total mass.")
        return -1
    
    print(f"CO core mass for remnant mass {remnant_mass} and total mass {total_mass}: {CO_core_mass}")
    plot(combined_core_mass, total_mass, remnant_mass, CO_core_mass=CO_core_mass)
    return CO_core_mass

def find_CO_core_mass(remnant_mass, total_mass, combined_core_mass):
    equation_to_solve = lambda core_mass: get_remnant_mass(core_mass, total_mass) - remnant_mass
    initial_guess = combined_core_mass
    solution = fsolve(equation_to_solve, initial_guess)
    if solution.size > 0:
        return solution[0]
    else:
        return -1

def core_mass_for_dataframe(df, n):
    df[f'sn_{n}_CO_core_mass'] = df.apply(lambda row: find_CO_core_mass(row[f'sn_{n}_remnant_mass'],
                                                                        row[f'sn_{n}_mass_{n}'],
                                                                        row[f'sn_{n}_massc_{n}']), axis=1)
    return df

def get_CO_cores(df):
    for n in [1,2]:
        df = core_mass_for_dataframe(df, n)
    return df