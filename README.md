---

## Hello!

This repository contains the supporting code for  
**“Properties of Core Collapse Supernovae from Binary Population Synthesis”**  
([placeholder arXiv link]).

It includes scripts, modules, and utilities developed which may be used to reproduce and extend the results from the paper. Specifically, the repository provides:

1. **Plotting scripts** to recreate all figures presented in the paper, including example notebooks.
2. **Supernova classification utilities**, enabling users to modify or extend the CCSN subtype schemes (see below).  
3. **Rerun scripts for COSMIC**, which include the detailed **common-envelope evolution (CEE)** prescription of  
   [Klencki et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A..54K/abstract).  

We also include the exact sampling and population-evolution scripts used to generate the original datasets. These scripts are provided **only as a procedural receipt**: they rely heavily on the specific directory structure used during our runs and will not function out of the box in a different environment. In most cases, **more efficient and general versions of this functionality are already implemented in the modules provided in this repository**. The original scripts are available in the `scripts/` directory.

---

## Data Access

To compile the plots shown in the accompanying notebooks and scripts, please download the processed **CCSN population dataframes** from the **Zenodo archive**:

[**Zenodo dataset placeholder link**](https://zenodo.org/placeholder)

These dataframes contain all binary systems that underwent one or more core-collapse supernovae in the COSMIC population grids. Each row corresponds to a unique binary (`bin_num`) and includes information about both the first and second supernova events.  

The datasets are structured with the following columns:

| Column | Description |
|--------|--------------|
| `bin_num` | Unique binary identifier within the COSMIC population. |
| `SN_1`, `SN_2` | Integer flag describing supernova (see COSMIC documentation for definitions). |
| `merger_type` | Merger classification if applicable (e.g., '13-13' for ns-ns, see COSMIC documentation). |
| `zams_mass_*` | ZAMS (Zero-Age Main Sequence) masses of the primary and secondary stars in solar masses. |
| `zams_porb` | Initial orbital period (days). |
| `zams_ecc` | Initial orbital eccentricity. |
| `zams_sep` | Initial orbital separation (R⊙). |
| `sn_*_time` | Time (in Myr) at which each supernova occurred. |
| `sn_*_mass_1`, `sn_*_mass_2` | Stellar masses (M⊙) at the time of each supernova. |
| `sn_*_massc_1`, `sn_*_massc_2` | Core masses (M⊙) at the time of each supernova. |
| `sn_*_menv_1`, `sn_*_menv_2` | Envelope masses (M⊙) at the time of each supernova. |
| `sn_*_kstar_1`, `sn_*_kstar_2` | Stellar type flags (`kstar`) for both stars (see COSMIC documentation for definitions). |
| `sn_*_porb`, `sn_*_ecc`, `sn_*_sep` | Orbital period, eccentricity, and separation immediately before each supernova. |
| `sn_*_max_loss_rate` | Maximum mass-loss rate (M⊙/yr) during the last kyr before the SN. |
| `sn_*_remnant_mass` | Compact-object remnant mass (M⊙) after the SN. |
| `sn_*_donor_kstars` | List or identifier of donor `kstar` types involved in binary interactions prior to the SN. |
| `sn_*_interactions` | Either None, 'RLOF', or 'CEE', corresponds to only pre-SN. |
| `sn_*_last_donor` | Boolean, whether the star was the donor in the last Pre-SN mass transfer. |
| `sn_*_merger` | Boolean indicating if the system merged before the SN. |
| `is_single` | Boolean flag for isolated single-star systems (at formation). |
| `sn_*_ns` | Boolean flags indicating whether the remnant produced is a neutron star. |
| `sn_*_CO_core_mass` | Carbon–oxygen core mass at collapse. |
| `sn_*_m_ejecta` | Total ejected mass (M⊙) during the SN. |
| `sn_*_m_h_ejecta`, `sn_*_m_he_ejecta`, `sn_*_m_co_ejecta` | Hydrogen, helium, and carbon/oxygen ejecta components (M⊙). |
| `sn_*_type` | Physical supernova type (either, 'I' or 'II', or some exotic type like 'ECSN'). |
| `sn_*_subtype` | Observational subtype (all systems with sn_*_type of 'I' or 'II' is assgined one of, IIn, IIP, IIL, IIb, Ib, Ic). |
| `sample_mass` | Total sampled stellar mass (M⊙) for population scaling. |
| `singles_mass` | Mass contribution from single-star evolution in the same model. |
| `sigma` | Natal-kick dispersion (km s⁻¹) population was evolved with. |
| `alpha1` | Common-envelope efficiency parameter α used for evolution (this column is set to 1.0  for simplicity in our custom models CEE, though this is not true). |
| `met_cosmic` | Metallicity (Z) used in COSMIC run. |

Here, all columns marked with an asterisk (`_*`) appear twice, once with a 1 for the **primary** and once with a 2 for the **secondary**. The supernova from the primary
star does not necessarily happen before the secondary! The order of the supernovae can be determined from the (`sn_*_time`) columns.

Due to storage limitations, the **initial COSMIC populations** and **full binary evolution histories** are *not included* in the Zenodo archive.  

However, you can easily regenerate any population by reconstructing the initial binaries from the `zams_` columns and evolving them using the `BSEDict` settings defined in  
[`scripts/setup_scripts/run_gen_pop.py`](scripts/setup_scripts/run_gen_pop.py).

This enables exact reproduction or re-evaluation of the binary population under updated assumptions.

---

## Supernova Classification

The supernova classification routines (see [`ccsnlab/sn_subtypes.py`](src/ccsnlab/sn_subtypes.py)) are designed to operate on dataframes structured as above.  
You may also rerun the classification with modified assumptions (e.g., subtype boundaries or mass thresholds) by using the provided functions and example Jupyter notebooks.

---

## Re-running Individual Binary Systems

The repository includes dedicated functionality to **extract and re-evolve individual binary systems** exactly as they were originally evolved in COSMIC.

- The module `data_loading.rerun_binary` provides tools to **pull out the initial conditions and all COSMIC evolution parameters** for any binary contained in the processed population dataframe.
- This module also includes helpers to **re-evolve the binary with the original BSEDict settings** and random seed.
- During this rerun, the code also performs all required bcm-array manipulations, particularly saving the **last kyr before core collapse** for each SN, facilitating the identification of **Type IIn** events.
- These reconstructed binaries may then be **re-evolved using the `cee_rerun` modules**. An example Jupyter notebook is provided that walks through the full workflow: extracting initial conditions, re-evolving the system as originally run, then re-evolving with the detailed CEE prescription of Klencki et al. (2021), and finally producing the `sn_info` dataframe used for analysis and plotting.

---
## Curly Brace Annotation

One of the plotting function in `plotting.subtypes` uses the external package  
[**matplotlib-curly-brace**](https://matplotlib-curly-brace.readthedocs.io/en/latest/installation_and_setup.html#)  
to draw curly braces in multi-panel figures.

If you wish to exactly replicate the plots, please install this package following the linked instructions.  
Otherwise, the inclusion of the curly brace can be toggled via a keyword argument in the plotting function.

---
