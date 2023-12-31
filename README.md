# blueshifts
[![DOI](https://zenodo.org/badge/694044692.svg)](https://zenodo.org/badge/latestdoi/694044692)
[![Tests](https://github.com/jhmatthews/blueshifts/actions/workflows/test_figures.yml/badge.svg)](https://github.com/jhmatthews/blueshifts/actions/workflows/test_figures.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2309.14434-b31b1b.svg)](http://arxiv.org/abs/2309.14434)

Scripts and supplementary data for the paper: 

_"A disc wind model for blueshifts in quasar broad emission lines"_, James H. Matthews, Jago Strong-Wright, Christian Knigge, Paul Hewett, Matthew J. Temple, Knox S. Long, Amy L. Rankine, Matthew Stepney, Manda Banerji, Gordon T. Richards, MNRAS, 2023, in press, [arXiv:2309.14434](http://arxiv.org/abs/2309.14434)

If you use any of this data please cite the paper accordingly. 

## Directory structure 

* **Data:** Data outputted from the radiative transfer calculations. 
* **Scripts:** Python scripts for making figures for the paper
* **supplement:** Supplementary material (source TeX and figures)
* **Figures:** Empty directory where the figures go
  
## Making figures 

To make figures, just type
```
python Scripts/make_all_figures.py
```

Requirements/dependencies are given in requirements.txt and can be installed via 

```
pip -r requirements.txt
```

## Data
The main simulation data is organised into folders for values of alpha (the acceleration exponent). Each simulation has the root filename of the following format:
```
runA_thminB_rvC_vinfD_mdotwE
```
where A is the run number, B is the inner wind angle, C is the acceleration length, vinf is the terminal velocity, E is the mass loss rate. Each run has the following files:

* **.pf**: parameter file (parameters are described [here](agnwinds.readthedocs.io/en/dev)
* **.spec**: spectrum output file. Can be read and plotted as descibed [here](https://agnwinds.readthedocs.io/en/dev/plotting/plot_spectrum.html)
* **.master.txt**: cell-by-cell data in the wind model. Can be read and plotted as descibed [here](https://agnwinds.readthedocs.io/en/dev/plotting/plot_wind.html)
* **.dat**: additional cell-by-cell data in the wind model that isn't in the master table.
