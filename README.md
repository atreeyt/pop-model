# Simple Population Model

- Description
- Installation
- Running the project

# Description

todo

# Installation
Only Python is required to run the project (version >3.7).

# Running the project
Entry point to the project is `pop_model.py`.
Use `population_model.PopulationModel()` to define the chromosomes in the population.
Use `events()` to adjust parameters for each time step, e.g. herbicide application that kills the susceptible individuals.

In order to run the default settings for four time steps, use the terminal to run:
```
python3 pop_model -t 4
```
The verbosity flag `-v` can be added to show additional information. `-vv` shows even more debugging information.
See `--help` for all command line arguments.