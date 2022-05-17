# Rutgers University Weather Research and Forecasting Model Validation

Scripts used for Validation of the RU-WRF model by comparing various outputs to observations points. Originally developed for Rutgers' RUWRF real-time model.

Author: Jaden Dicopoulos (jadend@marine.rutgers.edu)

Rutgers Center for Ocean Observing Leadership

## Installation Instructions
Add the channel conda-forge to your .condarc. You can find out more about conda-forge from their website: https://conda-forge.org/

`conda config --add channels conda-forge`

Clone the wrf_validation repository

`git clone https://github.com/JadenD/wrf_validation.git`

Change your current working directory to the location that you downloaded wrf_validation. 

`cd /Users/your_name/your_directory_structure/wrf_validation/`

Create conda environment from the included environment.yml file:

`conda env create -f environment.yml`

Once the environment is done building, activate the environment:

`conda activate wrf_val`

Install the toolbox to the conda environment from the root directory of the wrf_tools toolbox:

`pip install .`

The toolbox should now be installed to your conda environment.
