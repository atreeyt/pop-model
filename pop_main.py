import argparse
import logging
import os
from copy import deepcopy
from math import ceil

import observer_model
import population_model
import utils


def parse_arguments() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", help="max number of timesteps", dest="maxtime", type=int)
    parser.add_argument(
        "-l",
        "--log",
        help="Log to default file (logs/)",
        dest="use_logger",
        action="store_true",
    )
    parser.add_argument(
        "--log_file",
        help='Log file name. Should end with ".log". Default "basic.log".',
        dest="log_name",
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (use -v for verbose, -vv for more verbose)",
    )
    parser.add_argument(
        "--level",
        help=(
            "Set logging level"
            " [debug, info, warning, error, critical]"
            " (default warning), overrides -v"
        ),
        dest="log_level",
        default=None,
    )
    args = parser.parse_args()
    return args


def config(
    use_logger=False,
    log_name=None,
    log_folder=None,
    log_level="warning",
    verbose=0,
) -> None:
    """Defines any configuration for the file."""

    assert log_level in [
        "debug",
        "info",
        "warning",
        "error",
        "critical",
        None,
    ], f"Unknown logging level: {log_level}."

    match log_level:
        case "debug":
            log_level = logging.DEBUG
        case "info":
            log_level = logging.INFO
        case "warning":
            log_level = logging.WARNING
        case "error":
            log_level = logging.ERROR
        case "critical":
            log_level = logging.CRITICAL
        case _:
            if verbose == 0:
                log_level = logging.WARNING  # default
            elif verbose == 1:
                log_level = logging.INFO
            elif verbose == 2:
                log_level = logging.DEBUG
            elif verbose > 2:
                log_level = logging.DEBUG
                logging.warning("Verbosity set >2 has no effect.")

    if use_logger:
        if log_folder is None:
            log_folder = "logs"
        if log_name is None:
            log_name = "basic.log"
        # Create path logging directory.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, log_folder, log_name)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=f"{path}",
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    return


def get_resistant_seed_freq_from_pop(seed_dictionary, susceptible_list=["rr"]) -> float:
    """Get the frequency of the resistant seeds within the given population.

    Returns a float [0-1].
    """

    population = sum(seed_dictionary.values())
    total_count = 0
    for chromosome, count in seed_dictionary.items():
        if chromosome in susceptible_list:
            continue
        total_count += count
    return total_count / population


def get_resistant_seed_freq_from_freq(
    seed_freq_dictionary, susceptible_list=["rr"]
) -> float:
    """Given a dictionary of the frequency of each seed type within a
    population, returns the frequency of resistant seeds.

    Returns a float [0-1].
    """

    total_count = 0
    for chromosome, count in seed_freq_dictionary.items():
        if chromosome in susceptible_list:
            continue
        total_count += count
    return total_count


def get_population_history(
    iteration_history: list[population_model.PopulationModel], location
) -> list:
    """Get the population history of each iteration of the model.

    Returns a list containing absolute population values.
    """

    assert location in [
        "underground",
        "overground",
    ], f"location must be ['underground','overground']"

    return [
        sum(t.get_population(location=location).values()) for t in iteration_history
    ]


def get_resistance_history(
    iteration_history: list[population_model.PopulationModel], location
) -> list:
    """Get the resistance history of each iteration of the model.

    Returns a list containing resistance values.
    """
    resistance_history = [
        get_resistant_seed_freq_from_freq(t.get_frequency(location=location))
        for t in iteration_history
    ]
    return resistance_history


def calculate_year_from_t(t, time_steps_per_year) -> int:
    # math.ceil
    return ceil(t / time_steps_per_year)


def pretty_print_dict(dictionary, indent=0, use_tabs=False) -> None:
    for key, val in dictionary.items():
        char = "\t" if use_tabs else " "
        print(f"{char*indent}{key} : {val}")
    return


def print_population_stats(
    pop_model: population_model.PopulationModel, location, n_digits=3
) -> None:

    population = pop_model.get_population(location=location)
    print("    Population:")
    pretty_print_dict(utils.round_dict_values(population, n=n_digits), indent=8)
    print("        TOTAL :", round(sum(population.values()), 2))

    frequency = pop_model.get_frequency(location=location)
    print("    Frequency:")
    pretty_print_dict(utils.round_dict_values(frequency, n=n_digits), indent=8)
    print(
        "Frequency of resistant seeds:"
        f" {get_resistant_seed_freq_from_freq(frequency):.3f}"
    )
    return


def events(
    pop_model: population_model.PopulationModel, t: int
) -> population_model.PopulationModel:
    """Define events that occur at time t. Returns modified population model.

    At times t, certain events can happen to a population, such as
    culling a susceptible population. These changes occur here and the
    resultant population model is returned.

    year = math.ceil(t/(t per year))
    """
    # TODO add TIME_STEPS_PER_YEAR
    #   Calculate when to germinate, purge, and return seeds based on
    #   this value.
    match t:
        case 0:
            pop_model.add_seeds("rr", 100, location="underground")
            pop_model.add_seeds("Rr", 1, location="underground")

        # When 2t per year, all odd t are midyear.
        case t if not utils.is_even(t):
            pop_model.return_seeds_to_seedbank(rate=1.0)
            pop_model.germinate_seeds(rate=0.5)

        # When 2t per year, all even t are end of year.
        case t if utils.is_even(t):
            # An effective herbicide but only targeting susceptible individuals.
            pop_model.purge_population(0.8, ["rr"], location="overground")
            pop_model
        # Herbicide with less efficacy but different mode of action, all seeds susceptible.
        # pop_model.purge_population(
        #     0.4, ["rr", "rR", "Rr", "RR"], location="overground"
        # )
    return pop_model


def main(max_time=1) -> None:
    # List to store each iteration. Python passes the variable around as a
    # reference, so the object stored in the list will be modified after it
    # has been added.
    TIME_STEPS_PER_YEAR = 2
    iteration_history: list[population_model.PopulationModel] = []

    # Compute for number of time steps t.
    for t in range(0, max_time + 1):
        print(
            f"\n\n--- Time {t}, year {calculate_year_from_t(t,TIME_STEPS_PER_YEAR)} ---"
        )
        if t == 0:
            pop_model = population_model.PopulationModel()
        else:
            pop_model = deepcopy(iteration_history[t - 1])
        iteration_history.append(pop_model)

        # Modify population model with any 'events' such as adding seeds.
        pop_model = events(pop_model, t)

        # Model population changes after the start.
        if t > 0:
            results = pop_model.apply_population_change()

        # Showing results.
        print("  --OVERGROUND--")
        print_population_stats(pop_model, "overground")
        print("  --UNDERGROUND--")
        print_population_stats(pop_model, "underground")

    # Print lists of population and resistance history for graphs.
    # TODO matplotlib
    print("\npopulation history:")
    print(get_population_history(iteration_history, location="overground"))
    print("\nfrequency history:")
    print(get_resistance_history(iteration_history, location="overground"))

    # print("\n\n NOISY OBSERVER\n----------------")
    # observer = observer_model.ObserverModel(
    #     observation_accuracy=0.9, noise_standard_dev=0.05
    # )
    # for t, model in enumerate(iteration_history):
    #     print(f"\n\n--- Time {t} ---")
    #     # print("model.get_population():", model.get_population())
    #     population = observer.observe(model, noisy=True)
    #     print(f"  pop:")
    #     pretty_print_dict(utils.round_dict_values(population), indent=4)
    #     print("    TOTAL :", round(sum(population.values()), 2))
    #     #     ,
    #     #     f" total: {round(sum(population.values()),2):_}",
    #     # )
    #     population = model.get_population(location="overground")
    #     print("actual population:")
    #     print(
    #         f"  pop:",
    #         utils.round_dict_values(population),
    #         f" total: {round(sum(population.values()),2):_}",
    #     )

    return


if __name__ == "__main__":
    args: argparse.ArgumentParser = parse_arguments()
    config(
        use_logger=args.use_logger,
        log_name=args.log_name,
        log_level=args.log_level,
        verbose=args.verbose,
    )
    logging.debug("----- BEGIN PROGRAM -----")
    main(args.maxtime)
