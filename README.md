# Simple Population Model

- Description
- Installation
- Running the project

# Description

TODO

# Installation
Only Python is required to run the project (version >3.7).

# Running the project
- Entry point to the project is `pop_model.py`.
- Use `population_model.PopulationModel()` to define the chromosomes in the population.
    - There is an underground and an overground population.
    - The underground population germinates and becomes the overground population.
    - The overground population experiences the events, such as herbicide and reproduction.
    - The overground population returns to the underground population (the seedbank) at the end of the season.
    - Seeds within the underground population are safe from events.
    - Note: the underground population does not expire. (TODO?)
- Use `events()` to adjust parameters for each time step, e.g. herbicide application that kills the susceptible individuals.
    - Note how many timesteps are in a year, e.g. 1 year per timestep or 1 year per 2 timesteps (allows for summer and winter season).

In order to run the default settings for four time steps, use the terminal to run:
```
python3 pop_model -t 4
```
The verbosity flag `-v` can be added to show additional information. `-vv` shows even more debugging information.
See `--help` for all command line arguments.