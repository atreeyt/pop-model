import argparse
import logging
import os
from copy import deepcopy

import population_model
import utils


def parse_arguments() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', help='max number of timesteps', dest='maxtime', 
                        type=int)
    parser.add_argument('-l', '--log',
                        help='Log to default file (logs/)',
                        dest='use_logger', action='store_true')
    parser.add_argument('--log_file',
                        help='Log file name. Should end with ".log".'
                        ' Default "basic.log".',
                        dest='log_name', default=None)
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Increase verbosity level (use -v for verbose, -vv for more verbose)")
    parser.add_argument('--level', 
                        help='Set logging level'
                            ' [debug, info, warning, error, critical]'
                            ' (default warning), overrides -v', 
                        dest='log_level', default=None) 
    args = parser.parse_args()
    return args

def config(use_logger=False, log_name=None, log_folder=None, 
           log_level='warning', verbose=0) -> None:
    """Defines any configuration for the file."""

    assert log_level in [
        'debug', 'info', 'warning', 'error', 'critical', None
    ], f'Unknown logging level: {log_level}.'

    match log_level:
        case 'debug': 
            log_level = logging.DEBUG
        case 'info': 
            log_level = logging.INFO
        case 'warning': 
            log_level = logging.WARNING
        case 'error': 
            log_level = logging.ERROR
        case 'critical': 
            log_level = logging.CRITICAL
        case _:
            if args.verbose == 0:
                log_level = logging.WARNING # default
            elif args.verbose == 1: 
                log_level = logging.INFO
            elif args.verbose == 2:
                log_level = logging.DEBUG
            elif args.verbose > 2:
                log_level = logging.DEBUG
                logging.warning('Verbosity set >2 has no effect.')

    if use_logger:
        if log_folder is None:
            log_folder = 'logs'
        if log_name is None:
            log_name = 'basic.log'
        # Create path logging directory.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, log_folder, log_name)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
 
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=f'{path}'
        )
    else:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    return

def get_resistant_seed_freq_from_pop(seed_dictionary, 
                                           susceptible_list = ['rr']) -> float:
    population = sum(seed_dictionary.values())
    total_count = 0
    for chromosome, count in seed_dictionary.items():
        if chromosome in susceptible_list:
            continue
        total_count += count
    return total_count / population


def get_resistant_seed_freq_from_freq(seed_freq_dictionary,
                                            susceptible_list = ['rr']) -> float:
    total_count = 0
    for chromosome, count in seed_freq_dictionary.items():
        if chromosome in susceptible_list:
            continue
        total_count += count
    return total_count

def get_population_history(
        iteration_history: list[population_model.PopulationModel]) -> list:
    return [ sum(t.get_population().values()) for t in iteration_history ]

def get_resistance_history(
        iteration_history: list[population_model.PopulationModel]) -> list:
    [ get_resistant_seed_freq_from_freq(t.get_frequency()) for t in iteration_history ]


def events(pop_model: population_model.PopulationModel,
           t: int) -> population_model.PopulationModel:
    """Define events that occur at time t. Returns modified population model.

    At times t, certain events can happen to a population, such as culling
    a susceptible population. These changes occur here and the resultant
    population model is returned.
    """
    match t:
        case 0:
            pop_model.add_seeds('rr', 100_000)
            pop_model.add_seeds('Rr', 1)
        case t if t > 0:
            # Remove 80% of the 'rr' population.
            # TODO make this variable.
            count = pop_model.get_population()['rr']
            print('''Purging 80% of rr seeds.''')
            pop_model.remove_seeds('rr', count*0.8)

    return pop_model

def main(max_time=1) -> None:
    # List to store each iteration. Python passes the variable around as a
    # reference, so the object stored in the list will be modified after it
    # has been added.
    iteration_history = []

    # Compute for number of time steps t.
    for t in range(0, max_time+1):
        print(f'--- Time {t} ---')
        if t == 0:
            pop_model = population_model.PopulationModel()
        else:
            pop_model = deepcopy(iteration_history[t-1])
        iteration_history.append(pop_model)

        # Modify population model with any 'events' such as adding seeds.
        pop_model = events(pop_model, t)

        # Model population changes after the start.
        if t > 0:
            results = pop_model.get_population_change()

        # Showing results.
        population = pop_model.get_population()
        print(f'  pop:', utils.round_dict_values(population),
              ' total:', {round(sum(population.values()),2)})
        frequency = pop_model.get_frequency()
        print('  freq:', utils.round_dict_values(frequency))
        print(f'frequency of resistant seeds: {get_resistant_seed_freq_from_freq(frequency):.3f}')

        # Print lists of population and resistance history for graphs.
        # TODO matplotlib
        print(get_population_history(iteration_history))
        print(get_resistance_history(iteration_history))

    return

if __name__ == "__main__":
    args: argparse.ArgumentParser = parse_arguments()
    config(use_logger=args.use_logger,
           log_name=args.log_name,
           log_level=args.log_level,
           verbose=args.verbose)
    logging.debug('----- BEGIN PROGRAM -----')
    main(args.maxtime)
