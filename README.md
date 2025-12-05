

# ccsnlab

This repository contains the supporting code developed for [**“Properties of Core Collapse Supernovae from Binary Population Synthesis.”**](https://doi.org/10.48550/arXiv.2511.23285) It includes scripts and utilities which may be used to reproduce and extend the results from the paper. Specifically, the repository provides:

1. **Plotting scripts** to recreate all figures presented in the paper, including example notebooks.
2. **Supernova classification utilities**, enabling users to modify or extend the core collapse supernova (CCSN) subtype schemes (see below), and classify the supernovae from your own COSMIC runs.
3. **Rerun scripts for COSMIC**, which implement a detailed **common envelope evolution (CEE)** prescription based on the results of [Klencki et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A..54K/abstract).  

---

## Data Access & Structure

If you desire to view the original results from the paper, or exactly replicate our plots, please download the processed **CCSN population dataframes** from the [**Zenodo archive**](https://doi.org/10.5281/zenodo.17620853).

These dataframes contain all binary systems that underwent one or more CCSNe in our COSMIC population grids. Each row corresponds to a unique binary (`bin_num`) and includes information about both the first and second supernova events, and the system at zero age main sequence (ZAMS). You can also create these dataframes from your own COSMIC output using the tools in the [`ccsnlab.data_loading`](src/ccsnlab/data_loading) module. A more thorough description of this is in the next section.

All columns listed with an asterisk (`_*`) appear twice, once with a 1 for the **primary** and once with a 2 for the **secondary**. The 1 and 2 do **not** correspond to the order in which the events occured, consult the (`sn_*_time`) column to determine their order. The columns are as follows:

| Column | Description |
|--------|--------------|
| `bin_num` | Unique binary identifier within the COSMIC population. |
| `SN_1`, `SN_2` | Integer flag describing supernova (see COSMIC documentation for definitions). |
| `merger_type` | Merger classification if applicable (e.g., '0101' for two main sequence stars, see COSMIC documentation). |
| `zams_mass_*` | ZAMS masses of the primary and secondary (M⊙). |
| `zams_porb` | Initial orbital period (days). |
| `zams_ecc` | Initial orbital eccentricity. |
| `zams_sep` | Initial orbital separation (R⊙). |
| `sn_*_time` | Time (in Myr) at which each supernova occurred. |
| `sn_*_mass_1`, `sn_*_mass_2` | Stellar masses (M⊙) at the time of each supernova. |
| `sn_*_massc_1`, `sn_*_massc_2` | Core masses (M⊙) at the time of each supernova. |
| `sn_*_menv_1`, `sn_*_menv_2` | Envelope masses (M⊙) at the time of each supernova. |
| `sn_*_kstar_1`, `sn_*_kstar_2` | Stellar type flags (`kstar`) for both stars (see COSMIC documentation for definitions). |
| `sn_*_porb`, `sn_*_ecc`, `sn_*_sep` | Orbital period, eccentricity, and separation immediately before each supernova. |
| `sn_*_max_loss_rate` | Maximum mass loss rate (M⊙/yr) during the last kyr before the SN (values in this column implying mass loss are negative, taken from COSMIC's `deltam_*` column). |
| `sn_*_remnant_mass` | Compact object remnant mass (M⊙) after the SN. |
| `sn_*_donor_kstars` | List of `kstar` types in which this star was a donor concatenated into a string, (e.g. '1-2'). |
| `sn_*_interactions` | Either None, 'RLOF', or 'CEE', corresponds to only pre-SN. |
| `sn_*_last_donor` | Boolean, whether the star was the donor in the last Pre-SN mass transfer. |
| `sn_*_merger` | Boolean indicating if the system merged before the SN. |
| `is_single` | Boolean flag for isolated single-star systems (at formation). |
| `sn_*_ns` | Boolean flags indicating whether the remnant produced is a neutron star. |
| `sn_*_CO_core_mass` | Carbon–oxygen core mass at collapse. |
| `sn_*_m_ejecta` | Total ejected mass (M⊙) during the SN. |
| `sn_*_m_h_ejecta`, `sn_*_m_he_ejecta`, `sn_*_m_co_ejecta` | Hydrogen, helium, and carbon-oxygen ejecta components (M⊙). |
| `sn_*_type` | Broad supernova type (either, 'I' or 'II', or some exotic type like 'ECSN'). |
| `sn_*_subtype` | Supernova subtype (all systems with sn_*_type of 'I' or 'II' is assgined one of, IIn, IIP, IIL, IIb, Ib, Ic). |
| `sample_mass` | Total sampled stellar mass (M⊙) for population scaling. |
| `singles_mass` | Mass contribution from single stars (at formation) in the same model. |
| `sigma` | Natal kick dispersion (km s⁻¹) population was evolved with. |
| `alpha1` | Common envelope efficiency parameter α used for evolution (this column is set to 1.0  for simplicity in our custom models CEE, though this is not true). |
| `met_cosmic` | Metallicity (Z) used in COSMIC run. **Not** relative to solar metallicity. To scale to solar metallicity divide by Z⊙=0.02. |

---

## Supernova Classification & Compiling Dataframes

The supernova classification routines ([`ccsnlab.sn_types`](src/ccsnlab/sn_types.py)) are designed to operate on dataframes structured as described above. The classification assumptions can be customized in a number of ways. We provide a notebook [`notebooks/classification_variations.ipynb`](notebooks/classification_variations.ipynb) that walks through examples of how to modify these assumptions and how to analyze their impacts on the CCSN population demographics

You can create your own `sn_info` dataframe from COSMIC output using the tools in [`ccsnlab.data_loading`](src/ccsnlab/data_loading) -- an example notebook [`notebooks/evolve_binary_demo.ipynb`](notebooks/evolve_binary_demo.ipynb) walks through this process with a small COSMIC output!

There are a few caveats to using this for any COSMIC output. All that is required by the function to create the data is a `bpp` and `bcm` dataframe from COSMIC. However, there may be errors or unintended side effects if the data is not saved in a compatible way.

- In order to correctly identify whether an event is an iron core collapse, or some other exotic type of supernova, we require the last row of the `bcm` dataframe for each binary in order to access the `SN_*` column, as this is more challenging to do with the `bpp` alone.
- To correctly identify the progenitors of Type IIn CCSNe (and Ibn CCSNe in the future), we require the last kyr of the `bcm` before every supernova to be saved.
- COSMIC data which does not use the Delayed Fryer remnant mass prescription will have bugs and unintended side effects in creating the ejecta profile of each supernova. This is since COSMIC currently outputs core mass as a single combined value, without separating helium and carbon–oxygen components. In our tools, the final carbon–oxygen core mass is inferred via a root solver that inverts the Delayed Fryer remnant–mass prescription. Future development will add support for other remnant mass prescriptions.

We provide a few helper functions which wrap COSMIC's evolution functionality, automatically adressing these concerns, and these are used in our example notebook [`notebooks/evolve_binary_demo.ipynb`](notebooks/evolve_binary_demo.ipynb)!

---

## Re-running Individual Binary Systems from the Paper Dataset

Due to storage limitations, the initial COSMIC populations and full binary evolution histories are *not included* in the Zenodo archive. To ensure that others can replicate our results, and view the specific details of the evolution of each binary, the repository includes dedicated functionality to **extract and re-evolve individual binary systems** exactly as they were originally evolved in our data.

We provide an example notebook [`notebooks/rerun_a_binary_demo.ipynb`](notebooks/rerun_a_binary_demo.ipynb) that walks through the entire workflow. It uses the tools in [`ccsnlab.data_loading`](src/ccsnlab/data_loading) to extract the initial conditions and COSMIC evolution parameters for any system in the processed dataset. The system is evolved and processed into a `sn_info` dataframe.

---

## Using the Detailed CEE Prescription

We provide a notebook [`notebooks/detailed_CEE_demo.ipynb`](notebooks/detailed_CEE_demo.ipynb) which walks through how to apply our detailed CEE prescription. Currently, this prescription is implemented in a way such that it takes in the bpp and bcm of a dataframe which must be initially evolved in such a way that every CEE merges. The reasons for this are:

- When an initial population is evolved such that the first CEE always merges, low mass systems which have stars not indepently massive enought for core collapse might be able to merge and produce CCSNe, and will thus be saved. Since any other prescription will by definition produce fewer mergers at the first CEE, in order to see how this prescription would truly perform on the same initial population of stars, it is important to begin with as many candidate binaries as possible.

- These prescriptions cause the majority of CEE to merge. Our prescription is implemented by finding the first onset of CEE, and either forcing a merger or proceeding with the αλ from Klencki et al. (2021) in a survival candidate. This is applied in an arbitrary number of rounds, such that a system with multiple CEE episodes will have the prescription applied at each onset. While this is not terribly slow, it is not very fast, especially for large population of a few ten thousand CCSNe which exist in each of our populations. It is quicker to first rerun the population such that everything merges in one go, since only a few systems will survive, and this need not occur a lot.

The notebook does this process, it is easier to see than to explain.

---

## Plotting

We provide a notebook [`notebooks/make_paper_plots.ipynb`](notebooks/make_paper_plots.ipynb) which produces all of the plots exactly as they appear in the paper. The functions which create these plots are not very flexible. We provide a demonstration of the more customizable plotting functionality which is used to build these larger functions in the notebook [`notebooks/plotting_demo.ipynb`](notebooks/plotting_demo.ipynb). We also include a direct of the two qcflag prescriptions in [`notebooks/plotting_qcflag_comparison.ipynb`](notebooks/plotting_qcflag_comparison.ipynb), which we argue do not show enough difference to have an alternate version of most of these plots featured in the paper.

---

## Curly Brace Annotation

One of the plotting functions in [`ccsnlab.plotting.subtypes`](src/ccsnlab/plotting/subtypes.py) uses the external package [**matplotlib-curly-brace**](https://matplotlib-curly-brace.readthedocs.io/en/latest/installation_and_setup.html#) to draw curly braces in multi-panel figures.

If you wish to exactly replicate the plots, please install this package following the linked instructions. Otherwise, the inclusion of the curly brace can be toggled via a keyword argument in the plotting function.

---

## Scripts Used to Sample and Evolve the Paper Dataset

Also available in this repository are the exact sampling and evolution scripts used to generate the original datasets. These scripts are provided as a procedural receipt: they rely heavily on the specific directory structure used during our runs and will not function out of the box in a different environment. In most cases, more efficient and general versions of this functionality are already implemented in the modules provided in this repository. The original scripts are available in the `scripts/` directory. I do not recommend you use these yourself, since there are more efficient ways to do most of these things in COSMIC that I did not know about when I started!
