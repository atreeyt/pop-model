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
import numpy as np


def parse_arguments():
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
) -> list[float]:
    """Get the population history of each iteration of the model.

    Returns a list containing absolute population values.
    """

    assert location in [
        "underground",
        "overground",
    ], "location must be ['underground','overground']"

    return [
        sum(t.get_population(location=location).values()) for t in iteration_history
    ]


def get_resistance_history(
    iteration_history: list[population_model.PopulationModel],
    location,
    if_no_seeds_then_max=False,
) -> list[float]:
    """Get the resistance history of each iteration of the model.

    Returns a list containing resistance values.
    TODO update
    """
    resistance_history = []
    for t in iteration_history:
        freq = t.get_frequency(location=location)
        resistant_freq = get_resistant_seed_freq_from_freq(freq)
        resistance_history.append(resistant_freq)
    return resistance_history


def get_t_from_month_year(MONTH, TIME_STEPS_PER_YEAR, YEAR=None) -> list:
    """TODO

    TODO this is horribly inefficient
    """

    list_of_t = []
    for t in TIME_STEPS_PER_YEAR:
        year = get_year(t, TIME_STEPS_PER_YEAR)
        month = get_month(t, TIME_STEPS_PER_YEAR)
        # Only return t when year and month.
        if YEAR:
            if year == YEAR and month == MONTH:
                return [t]
        else:
            if month == MONTH:
                list_of_t.append(t)
    return list_of_t


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
        case _:
            month = "January"
            logging.error(f"Month {month} is out of bounds. Returning 'January'.")
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


def calculate_percent_change(values):
    """TODO remove. Unused."""
    return [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]


def calculate_survival_rates(observations: dict) -> list:
    initial_pop = None
    survivor_pop = None
    survival_rate = None
    survival_rates = []
    for key, val in observations.items():
        _, month = key.split("-")
        month = int(month)
        if Month(month) == Month.OCT:
            initial_pop = float(val)
            survivor_pop = None
        if Month(month) == Month.MAR:
            survivor_pop = float(val)
        if initial_pop and survivor_pop:
            survival_rate = survivor_pop / initial_pop
            survival_rates.append(survival_rate)
            survival_rate = None

    logging.debug("survival_rates=", survival_rates)
    return survival_rates


def calculate_survival_rates_from_list(observations: list) -> list:
    """untested"""
    initial_pop = None
    survivor_pop = None
    survival_rate = None
    survival_rates = []
    for val in observations:
        if initial_pop is None:
            initial_pop = val
        elif survivor_pop is None:
            survivor_pop = val
        if initial_pop and survivor_pop:
            survival_rate = survivor_pop / initial_pop
            survival_rates.append(survival_rate)
            initial_pop = None
            survivor_pop = None
    return survival_rates


def trenddetector(list_of_index, array_of_data, order=1) -> float:
    """TODO remove. Unused"""
    result = np.polyfit(list_of_index, list(array_of_data), order)
    # print("trenddetector result=")
    # print(result)
    slope = result[-2]
    return float(slope)


def calculate_moving_averages(x: list, window_size: int = 3) -> list:
    moving_averages = []
    for i in range(0, len(x) - window_size + 1):
        window = x[i : i + window_size]  # Elements in the list to consider.
        window_average = sum(window) / window_size
        moving_averages.append(window_average)
    return moving_averages


def show_pop_and_res_graph(iteration_history, MAX_TIME, TIME_STEPS_PER_YEAR) -> None:
    _, ax1 = plt.subplots()

    x = pd.date_range(
        start="12/2000", periods=MAX_TIME * TIME_STEPS_PER_YEAR + 1, freq="ME"
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
    # y3 = get_resistance_history(iteration_history, location="overground")
    # y4 = get_resistance_history(iteration_history, location="underground")
    # ax2.plot(x, y3, "g:")
    # ax2.plot(x, y4, "r:")
    overall_pop = {}
    y_combined_resistance = []
    for t in iteration_history:
        underground_pop = t.get_population(location="underground")
        overground_pop = t.get_population(location="overground")
        for key in underground_pop:
            if key in overground_pop:
                overall_pop[key] = underground_pop[key] + overground_pop[key]
        resistance = get_resistant_seed_freq_from_pop(overall_pop)
        y_combined_resistance.append(resistance)
    # y_combined_resistance = [(a + b) / 2 for a, b in zip(y3, y4)]
    ax2.plot(x, y_combined_resistance, "b:")
    ax2.set_ylabel("resistance rate (...)")
    ax2.set_ylim((0.0, 1.0))

    ax1.legend()
    # ax2.legend()
    vlines = pd.date_range(start="12/2000", periods=MAX_TIME + 1, freq="YE")
    ax2.vlines(vlines, 0, 1, color="grey", linewidth=0.4, alpha=0.8)

    # Extra vlines for displaying events within the first year (for visual clarity).
    # vlines = pd.date_range(start="02/2025", periods=1, freq="D")
    # ax2.vlines(
    #     vlines,
    #     0,
    #     1,
    #     color="red",
    #     linestyles="dashed",
    #     linewidth=1,
    #     alpha=0.8,
    #     label="germination",
    # )

    # vlines = pd.date_range(start="06/2025", periods=1, freq="D")
    # ax2.vlines(
    #     vlines,
    #     0,
    #     1,
    #     color="blue",
    #     linestyles="dashed",
    #     linewidth=1,
    #     alpha=0.8,
    #     label="crossing",
    # )

    # vlines = pd.date_range(start="08/2025", periods=1, freq="D")
    # ax2.vlines(
    #     vlines,
    #     0,
    #     1,
    #     color="green",
    #     linestyles="dashed",
    #     linewidth=1,
    #     alpha=0.8,
    #     label="seeds return",
    # )

    # vlines = pd.date_range(start="10/2025", periods=1, freq="D")
    # ax2.vlines(
    #     vlines,
    #     0,
    #     1,
    #     color="purple",
    #     linestyles="dashed",
    #     linewidth=1,
    #     alpha=0.8,
    #     label="germination",
    # )

    # vlines = pd.date_range(start="11/2025", periods=1, freq="D")
    # ax2.vlines(
    #     vlines,
    #     0,
    #     1,
    #     color="orange",
    #     linestyles="dashed",
    #     linewidth=1,
    #     alpha=0.8,
    #     label="herbicide applied",
    # )
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
        # Herbicide resistant biotypes occur at frequencies of 10e-8
        # or less (100_000_000). Simard and Laforest, 2024, p. 533.
        pop_model.add_seeds("rr", 100_000_000, location="underground")
        change_occurred = True
        return pop_model, change_occurred

    if year == 15 and month == 2:
        # Do some event only needed for this month.
        pop_model.add_seeds("Rr", 10, location="underground")
        change_occurred = True
        # return pop_model, change_occurred

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


def main(MAX_TIME=1, verbose=False) -> None:
    # If this value is changed, events() must be changed too.
    TIME_STEPS_PER_YEAR = 12
    iteration_history: list[population_model.PopulationModel] = []
    observation_history: dict = {}

    observer = observer_model.ObserverModel(
        observation_accuracy=1.0, noise_standard_dev=0.0
    )

    # Compute for t amount of years.
    for t in range(0, MAX_TIME * TIME_STEPS_PER_YEAR + 1):
        logging.debug(f"timestep={t}")
        year = get_year(t, TIME_STEPS_PER_YEAR)
        month = get_month(t, TIME_STEPS_PER_YEAR)
        if verbose:
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
            # observation_history.append(observation)
            observation_history[f"{year}-{month}"] = observation
            if verbose:
                print("\tObservation:", observation)

    for year in range(2, MAX_TIME + 1):
        # Ignore first March and last October, as no pre/post to compare to.
        month = Month.MAR.value
        count_last_year = observation_history[f"{year-1}-{month}"]
        count_this_year = observation_history[f"{year}-{month}"]
        if count_this_year > count_last_year:
            print("Resistance detected using post-control, year", year)

    # Show graphs.
    # Population and resistance graph.
    show_pop_and_res_graph(iteration_history, MAX_TIME, TIME_STEPS_PER_YEAR)

    # Observation graph.
    plt.plot(
        [(index + 1) * 0.5 for index, val in enumerate(observation_history)],
        list(observation_history.values()),
        marker="o",
        linestyle="-",
    )
    plt.xlabel("year")
    plt.ylabel("observed population")
    plt.title("observed pop vs time")
    plt.show()
    # print("observation_history=", observation_history)
    # Survival rates graph.
    # The first observations in March is ignored, as no pre-control
    #     observation is available.
    survival_rates = calculate_survival_rates(observation_history)

    # Survival rates are measured starting from the second year,
    #     therefore +2. Mar 2nd year - Oct 1st year.
    plt.plot(range(2, len(survival_rates) + 2), survival_rates)
    plt.xlabel("year (summer), e.g. 2 is March 2002")
    plt.ylabel("rate")
    plt.title("survival rates")
    plt.show()

    # percent_changes = [
    #     round(i * 100, 4) for i in calculate_percent_change(survival_rates)
    # ]

    # Moving averages graph.
    window_size = 5
    moving_averages = calculate_moving_averages(survival_rates, window_size=window_size)

    plt.plot(range(window_size, len(survival_rates) + 1), moving_averages)
    plt.xlabel("time")
    plt.ylabel(f"average rate, window {window_size}")
    plt.title("moving averages of survival_rates")
    plt.show()

    # # Polynomial fitting (linear fit), regression analysis graph.
    # x = range(1, len(moving_averages) + 1)
    # y = moving_averages
    # coefficients = np.polyfit(x, y, 1)
    # p = np.poly1d(coefficients)
    # plt.scatter(x, y, label="Data Points")
    # plt.plot(x, p(x), label="Linear Fit", color="red")
    # plt.title('moving averages with linear fit')
    # plt.legend()
    # plt.show()

    # Graph moving averages+ linear fit, increasing number of points each iteration.
    coefficients_list = []
    for i in range(1, len(moving_averages)):
        x = range(window_size, i + window_size)
        y = moving_averages[:i]
        coefficients = np.polyfit(x, y, 1)
        coefficients_list.append(coefficients[0])
        p = np.poly1d(coefficients)
        plt.scatter(x, y, label="Data Points")
        plt.plot(x, p(x), label="Linear Fit", color="red")
        ax = plt.gca()
        # ax.set_xlim([xmin, xmax])
        ax.set_ylim([0.0, 1.5])
        plt.title("moving_averages with linear fit")
        plt.legend()
        plt.show()

    # Coefficients graph from the survival rates, this is the value that would be given in real time.
    print("coefficients_list (moving averages)=", coefficients_list)
    x = range(window_size, len(coefficients_list) + window_size)
    y = coefficients_list
    plt.plot(x, y)
    plt.title("coefficients from moving averages of survival rates")
    plt.show()

    # Graph survival rates + linear fit, increasing number of points each iteration.
    coefficients_list = []
    for i in range(1, len(survival_rates)):
        x = range(1, i + 1)
        y = survival_rates[:i]
        coefficients = np.polyfit(x, y, 1)
        coefficients_list.append(coefficients[0])
        p = np.poly1d(coefficients)
        plt.scatter(x, y, label="Data Points")
        plt.plot(x, p(x), label="Linear Fit", color="red")
        ax = plt.gca()
        ax.set_ylim([0.0, 1.5])
        plt.title("survival_rates with linear fit")
        plt.legend()
        plt.show()

    # Coefficients graph from the survival rates, this is the value that would be given in real time.
    print("coefficients_list (survival_rates)=", coefficients_list)
    plt.plot(range(1, len(coefficients_list) - 2), coefficients_list[3:])
    plt.title("coefficients from survival rates")
    plt.show()

    return


if __name__ == "__main__":
    args = parse_arguments()
    config(
        use_logger=args.use_logger,
        log_name=args.log_name,
        log_level=args.log_level,
        verbose=args.verbose,
    )
    logging.debug("----- BEGIN PROGRAM -----")
    main(MAX_TIME=args.maxtime, verbose=args.verbose)
