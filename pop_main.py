import argparse
import logging
import os
from copy import deepcopy
from enum import Enum
from math import ceil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

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
            elif verbose == 2:
                log_level = logging.INFO
            elif verbose == 3:
                log_level = logging.DEBUG
            elif verbose > 3:
                log_level = logging.DEBUG
                logging.warning("Verbosity set >3 has no effect.")

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


class Month(Enum):
    JAN = 1
    FEB = 2
    MAR = 3
    APR = 4
    MAY = 5
    JUN = 6
    JUL = 7
    AUG = 8
    SEP = 9
    OCT = 10
    NOV = 11
    DEC = 12


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


def get_year(t, time_steps_per_year) -> int:
    # math.ceil
    return ceil(t / time_steps_per_year)


def get_month(t, time_steps_per_year) -> int:
    """Get the 'month' of the year. Returns int.

    The 'month' has a max of time_steps_per_year. If this value is 4
    then the 'months' in the year are 1,2,3,4 and actually correlate to
    three real months each.
    """

    if t == 0:
        return 1
    month = t % time_steps_per_year
    return time_steps_per_year if month == 0 else month


def get_month_name(month_num) -> str:
    match month_num:
        case 1:
            month = "January"
        case 2:
            month = "February"
        case 3:
            month = "March"
        case 4:
            month = "April"
        case 5:
            month = "May"
        case 6:
            month = "June"
        case 7:
            month = "July"
        case 8:
            month = "August"
        case 9:
            month = "September"
        case 10:
            month = "October"
        case 11:
            month = "November"
        case 12:
            month = "December"
    return month


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
        "        Frequency of resistant seeds:"
        f" {get_resistant_seed_freq_from_freq(frequency)}"
    )
    return


def show_pop_and_res_graph(iteration_history, MAX_TIME, TIME_STEPS_PER_YEAR) -> None:
    fig, ax1 = plt.subplots()

    x = pd.date_range(
        start="12/2024", periods=MAX_TIME * TIME_STEPS_PER_YEAR + 1, freq="ME"
    )

    y1 = get_population_history(iteration_history, location="overground")
    y2 = get_population_history(iteration_history, location="underground")
    ax1.plot(x, y1, label="overground", color="mediumseagreen")
    ax1.plot(x, y2, label="underground", color="brown", linewidth=1.0, alpha=1.0)
    ax1.set_xlabel("year")
    ax1.set_ylabel("population (-)")
    ax1.set_ylim(bottom=0)

    ax1.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks every year
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Show only the year

    ax2 = ax1.twinx()
    y3 = get_resistance_history(iteration_history, location="overground")
    y4 = get_resistance_history(iteration_history, location="underground")
    ax2.plot(x, y3, "g:")
    ax2.plot(x, y4, "r:")
    ax2.set_ylabel("resistance rate (...)")
    ax2.set_ylim([0, 1])

    ax1.legend()
    # ax2.legend()
    vlines = pd.date_range(start="12/2024", periods=MAX_TIME + 1, freq="YE")
    ax2.vlines(vlines, 0, 1, color="grey", linewidth=0.4, alpha=0.8)

    # Extra vlines for displaying events within the first year (for visual clarity).
    vlines = pd.date_range(start="02/2025", periods=1, freq="D")
    ax2.vlines(
        vlines,
        0,
        1,
        color="red",
        linestyles="dashed",
        linewidth=1,
        alpha=0.8,
        label="germination",
    )

    vlines = pd.date_range(start="06/2025", periods=1, freq="D")
    ax2.vlines(
        vlines,
        0,
        1,
        color="blue",
        linestyles="dashed",
        linewidth=1,
        alpha=0.8,
        label="crossing",
    )

    vlines = pd.date_range(start="08/2025", periods=1, freq="D")
    ax2.vlines(
        vlines,
        0,
        1,
        color="green",
        linestyles="dashed",
        linewidth=1,
        alpha=0.8,
        label="seeds return",
    )

    vlines = pd.date_range(start="10/2025", periods=1, freq="D")
    ax2.vlines(
        vlines,
        0,
        1,
        color="purple",
        linestyles="dashed",
        linewidth=1,
        alpha=0.8,
        label="germination",
    )

    vlines = pd.date_range(start="11/2025", periods=1, freq="D")
    ax2.vlines(
        vlines,
        0,
        1,
        color="orange",
        linestyles="dashed",
        linewidth=1,
        alpha=0.8,
        label="herbicide applied",
    )
    ax2.legend(loc="upper center")
    plt.show()
    return


def events(
    pop_model: population_model.PopulationModel, t: int, TIME_STEPS_PER_YEAR: int
):  # -> tuple[population_model.PopulationModel, bool]:
    """Define events that occur at time t. Returns modified tuple[population model, bool].

    At times t, certain events can happen to a population, such as
    culling a susceptible population. These changes occur here and the
    resultant population model is returned.
    bool returned is if the population model was modified/changed at all.
    """
    # TODO fix docstring, e.g. return type
    change_occurred = False
    year = get_year(t, TIME_STEPS_PER_YEAR)
    month = get_month(t, TIME_STEPS_PER_YEAR)

    if t == 0:
        pop_model.add_seeds("rr", 1_000, location="underground")
        pop_model.add_seeds("Rr", 1, location="underground")
        change_occurred = True
        return pop_model, change_occurred

    # if year == 1 and month == 1:
    # Do some event only needed for this month.
    # return

    match Month(month):
        case Month.JAN:
            pass
        case Month.FEB:
            pop_model.germinate_seeds(rate=0.7)
            change_occurred = True
        case Month.MAR:
            pass
        case Month.APR:
            pass
        case Month.MAY:
            pass
        case Month.JUN:
            pop_model.apply_population_change()
            change_occurred = True
        case Month.JUL:
            pass
        case Month.AUG:
            pop_model.return_seeds_to_seedbank(rate=1.0)
            change_occurred = True
        case Month.SEP:
            pass
        case Month.OCT:
            pop_model.germinate_seeds(rate=0.7)
            change_occurred = True
        case Month.NOV:
            # An effective herbicide but only targeting susceptible individuals.
            pop_model.purge_population(0.8, ["rr"], location="overground")
            change_occurred = True
        case Month.DEC:
            pass

        # Herbicide with less efficacy but different mode of action, all seeds susceptible.
        # pop_model.purge_population(
        #     0.4, ["rr", "rR", "Rr", "RR"], location="overground"
        # )
    return pop_model, change_occurred


def main(MAX_TIME=1) -> None:
def main(MAX_TIME=1, verbose=False) -> None:
    TIME_STEPS_PER_YEAR = 12  # If this value is changed, events() must be changed too.
    iteration_history: list[population_model.PopulationModel] = []
    observation_history: list[float] = []

    observer = observer_model.ObserverModel(
        observation_accuracy=0.9, noise_standard_dev=0.05
    )

    # Compute for t amount of years.
    for t in range(0, MAX_TIME * TIME_STEPS_PER_YEAR + 1):
        logging.debug(f"timestep={t}")
        year = get_year(t, TIME_STEPS_PER_YEAR)
        month = get_month(t, TIME_STEPS_PER_YEAR)
        print(f"\n\n--- Year {year}, {get_month_name(month)} ---")
        if t == 0:
            pop_model = population_model.PopulationModel()
        else:
            pop_model = deepcopy(iteration_history[t - 1])
        iteration_history.append(pop_model)

        # This space here is 'start of year', before events.

        # Modify population model with any 'events' such as adding seeds.
        pop_model, change_occurred = events(pop_model, t, TIME_STEPS_PER_YEAR)

        # This space here is 'end of year', after events.
        # Model population changes after the start.
        # if (t > 0) and (t % TIME_STEPS_PER_YEAR == 0):
        #     results = pop_model.apply_population_change()

        # Showing results.
        if verbose:
            if change_occurred:
                print("  --OVERGROUND--")
                print_population_stats(pop_model, "overground")
                print("  --UNDERGROUND--")
                print_population_stats(pop_model, "underground")
            else:
                print("...")

        if Month(month) == Month.MAR or Month(month) == Month.OCT:
            # TODO store as observation data type with date and count?
            observation = observer.observe(pop_model, noisy=True)
            observation_history.append(observation)
            if verbose:
                print(f"\tObservation:", observation)

    # Print lists of population and resistance history for graphs.
    # show_pop_and_res_graph(iteration_history, MAX_TIME, TIME_STEPS_PER_YEAR)

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
    main(MAX_TIME=args.maxtime, verbose=args.verbose)
