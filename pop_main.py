import argparse
import logging
import os
from copy import deepcopy
from enum import Enum
from math import ceil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import observer_model
import pandas as pd
import population_model
import seaborn as sns
import utils


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t",
        "--time",
        help="Number of years to run the simulation for.",
        type=int,
    )
    parser.add_argument(
        "-n",
        "--noise",
        type=float,
        help="Noise standard deviation (float).",
        default=0.0,
    )
    parser.add_argument(
        "-a",
        "--accuracy",
        type=float,
        help="Observation accuracy (float). Default 1.0.",
        default=1.0,
    )
    parser.add_argument(
        "--fpr",
        type=float,
        help="False positive rate of observer (percentage). Default 0.0.",
        default=0.0,
        dest="fpr",
    )
    parser.add_argument(
        "-s",
        "--slow",
        action="store_true",
        help="Require confirmation to progress to the next year.",
    )
    parser.add_argument(
        "-l",
        "--log",
        help="Log to default file (logs/).",
        dest="use_logger",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (use -v for verbose, -vv for more verbose).",
    )
    args = parser.parse_args()
    return args


def config(
    use_logger=False,
    verbose=0,
) -> None:
    """Defines any configuration for the file."""

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
        log_folder = "logs"
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
    iteration_history: list[population_model.PopulationModel], location
) -> list[float]:
    """Get the resistance history of each iteration of the model.

    Returns a list containing resistance values.
    """
    resistance_history = []
    for t in iteration_history:
        freq = t.get_frequency(location=location)
        resistant_freq = get_resistant_seed_freq_from_freq(freq)
        resistance_history.append(resistant_freq)
    return resistance_history


def get_t_from_month_year(MONTH, TIME_STEPS_PER_YEAR, YEAR=None) -> list:
    """Return the timesteps that correlates to MONTH.

    If YEAR, return a list with a single element of MONTH and YEAR.

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
    """Return the year that timestep t belongs to."""
    return ceil(t / time_steps_per_year)


def get_month(t, time_steps_per_year) -> int:
    """Get the 'month' of the year. Returns int.

    The 'month' has a max of time_steps_per_year. If this value is 4
    then the 'months' in the year are 1,2,3,4 and actually correlate to
    three real months each, e.g. '1' is January, February, and March.
    """

    if t == 0:
        return 1
    month = t % time_steps_per_year
    return time_steps_per_year if month == 0 else month


def get_month_name(month_num) -> str:
    """Returns the month's string name given its numerical value [1-12].

    Out of bounds cases return 'January'.
    """
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
        print(f"{char * indent}{key} : {val}")
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


def calculate_percent_change(values: list[float | int]):
    """Calculate the percentage change between i and i-1 in a list of values.

    Returns a list of survival rates. These rates can be negative.
    len(survival_rates) = len(values) - 1
    """
    return [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]


def calculate_survival_rates(observations: dict) -> list:
    """Calculate the survival rates from the observation dictionary."""
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


def calculate_moving_averages(x: list, window_size: int = 3) -> list:
    moving_averages = []
    for i in range(0, len(x) - window_size + 1):
        window = x[i : i + window_size]  # Elements in the list to consider.
        window_average = sum(window) / window_size
        moving_averages.append(window_average)
    return moving_averages


def calculate_aic(n, sse, k) -> float:
    """Calculate akaike information criterion. Returns float."""
    aic = n * np.log(sse / n) + 2 * k
    return aic


def calculate_bic(n, sse, k) -> float:
    """Calculate bayesian information criterion. Returns float."""
    bic = n * np.log(sse / n) + k * np.log(n)
    return bic


def calculate_sse(y, y_pred) -> list:
    """Calculate sum of squared errors. Return list of floats.

    y and y_pred are array-like.
    """
    sse = np.sum((y - y_pred) ** 2)
    return sse


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
    # ax2.legend(loc="upper center")
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
    change_occurred = False
    year = get_year(t, TIME_STEPS_PER_YEAR)
    month = get_month(t, TIME_STEPS_PER_YEAR)

    if t == 0:
        # Herbicide resistant biotypes occur at frequencies of 10e-8
        # or less (100_000_000). Simard and Laforest, 2024, p. 533.
        pop_model.add_seeds("rr", 100_000_000, location="underground")
        change_occurred = True
        return pop_model, change_occurred

    if year == 1 and Month(month) == Month.FEB:
        # Do some event only needed for this month.
        pop_model.add_seeds("Rr", 1_000, location="underground")
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


def main(args) -> None:
    MAX_TIME = args.time
    VERBOSE = args.verbose
    SLOW = args.slow
    NOISE_STD_DEV = args.noise  # depreciated. TODO remove.
    OBSERVATION_ACCURACY = args.observation_accuracy
    OBSERVATION_FPR = args.fpr

    # If this value is changed, events() must be changed too.
    TIME_STEPS_PER_YEAR = 12
    iteration_history: list[population_model.PopulationModel] = []
    observation_history: dict = {}

    observer = observer_model.ObserverModel(
        accuracy=OBSERVATION_ACCURACY,
        fpr=OBSERVATION_FPR,
        noise_standard_dev=NOISE_STD_DEV,
    )

    last_year = 0
    # Compute for t amount of years.
    for t in range(0, MAX_TIME * TIME_STEPS_PER_YEAR + 1):
        logging.debug(f"timestep={t}")
        year = get_year(t, TIME_STEPS_PER_YEAR)
        # TODO Why compute this every time? Just +1 until new year?
        month = get_month(t, TIME_STEPS_PER_YEAR)
        if VERBOSE:
            # Every year pause and wait for confirmation to continue.
            if SLOW and year > last_year:
                input("Press ENTER to continue...")
                last_year = year
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

        # Showing results.
        if VERBOSE:
            if change_occurred:
                print("  --OVERGROUND--")
                print_population_stats(pop_model, "overground")
                print("  --UNDERGROUND--")
                print_population_stats(pop_model, "underground")
            else:
                print("...")

        if Month(month) == Month.MAR or Month(month) == Month.OCT:
            observation = observer.observe(pop_model, noisy=True)
            # observation_history.append(observation)
            observation_history[f"{year}-{month}"] = observation
            if VERBOSE:
                print("\tObservation:", observation)

    for year in range(2, MAX_TIME + 1):
        # Ignore first March and last October, as no pre/post to compare to.
        month = Month.MAR.value
        count_last_year = observation_history[f"{year - 1}-{month}"]
        count_this_year = observation_history[f"{year}-{month}"]
        if count_this_year > count_last_year:
            print("Resistance detected using post-control, year", year)

    # Show graphs.
    # Population and resistance graph.

    # Print lists of population and resistance history for graphs.
    show_pop_and_res_graph(iteration_history, MAX_TIME, TIME_STEPS_PER_YEAR)
    print(observation_history)
    # Observation graph.
    plt.plot(
        # range(1, len(observation_history) + 1),
        [(index + 1) * 0.5 for index, _ in enumerate(observation_history)],
        list(observation_history.values()),
        marker="o",
        linestyle="-",
    )
    plt.xlabel("year")
    plt.ylabel("observed population")
    plt.title("observed population vs time")
    y1 = observation_history["1-10"]
    plt.hlines(
        y1,
        1,
        15,
        color="red",
        linestyles="dashed",
        linewidth=1,
        alpha=0.8,
        label=f"{y1}",
    )
    plt.text(15.2, y1, f"{y1:,.0f}", ha="left", va="center", c="red")
    y2 = observation_history["2-3"]
    plt.hlines(
        y2,
        1,
        15,
        color="red",
        linestyles="dashed",
        linewidth=1,
        alpha=0.8,
        label=f"{y2}",
    )
    plt.text(15.2, y2, f"{y2:,.0f}", ha="left", va="center", c="red")
    plt.arrow(
        1.8,  # x
        y1,  # y
        0,  # x step
        y2 - y1,  # y step
        head_width=0.3,
        head_length=3_000_000,
        linewidth=2,
        color="r",
        length_includes_head=True,
    )

    y1 = observation_history["8-10"]
    plt.hlines(
        y1,
        8,
        15,
        color="green",
        linestyles="dashed",
        linewidth=1,
        alpha=0.8,
        label="",
    )
    plt.text(15.2, y1, f"{y1:,.0f}", ha="left", va="center", c="green")
    y2 = observation_history["9-3"]
    plt.hlines(y2, 8, 15, color="green", linestyles="dashed", linewidth=1, alpha=0.8)
    plt.text(15.2, y2, f"{y2:,.0f}", ha="left", va="center", c="green")
    plt.arrow(
        8.8,  # x
        y1,  # y
        0,  # x step
        y2 - y1,  # y step
        head_width=0.3,
        head_length=3_000_000,
        linewidth=2,
        color="green",
        length_includes_head=True,
    )
    plt.savefig("test2.pdf", format="pdf")
    plt.show()

    # SURVIVAL RATES GRAPH.
    # The first observations in March is ignored, as no pre-control
    #     observation is available.
    survival_rates = calculate_survival_rates(observation_history)
    print("Graph of survival rates.")
    # Survival rates are measured starting from the second year,
    #     therefore +2. Mar 2nd year - Oct 1st year.
    plt.plot(range(2, len(survival_rates) + 2), survival_rates)
    plt.xlabel("year (summer), e.g. 2 is March 2002")
    plt.ylabel("rate")
    plt.title("survival rates")
    plt.show()

    # Surival rates LINEAR FIT.
    x = range(2, len(survival_rates) + 2)
    y = survival_rates
    coefficients = np.polyfit(x, y, 1)
    print("Survival rates Linear Fit Coefficients:", coefficients)
    # Create polynomial function
    p = np.poly1d(coefficients)
    plt.scatter(x, y, label="Data Points")
    plt.plot(x, p(x), label="Linear Fit", color="red")
    plt.legend()
    plt.title("Linear fit of survival rates")
    plt.show()

    # Graph survival rates + linear fit, increasing number of points each iteration.
    print("Graphs of survival rates with a linear fit.")
    coefficients_list = []
    for i in range(1, len(survival_rates) + 1):
        x = range(2, i + 2)
        y = survival_rates[:i]
        coefficients = np.polyfit(x, y, 1)
        coefficients_list.append(coefficients[0])
        p = np.poly1d(coefficients)
        plt.scatter(x, y, label="Data Points")
        plt.plot(x, p(x), label="Linear Fit", color="red")
        ax = plt.gca()
        ax.set_ylim((0.0, 1.5))
        plt.title("survival_rates with linear fit")
        plt.legend()
        print(coefficients_list[-1])
        plt.show()

    # Coefficients graph from the survival rates, this is the value that would be given in real time.
    print("Graph of coefficients from survival rates.")
    x = range(2, len(coefficients_list) + 2)
    y = coefficients_list
    plt.plot(x, y)
    plt.title("coefficients from survival rates")
    plt.show()

    # BOXPLOT OF SURVIVAL RATES.
    # Survival rates starts at 2, ends at MAX_TIME. Index 0 = 2. i = year-2.
    sns.boxplot(survival_rates[: 25 - 2])
    plt.title("boxplot of survival rates")
    plt.show()

    # MOVING AVERAGES.
    window_size = 5
    moving_averages = calculate_moving_averages(survival_rates, window_size=window_size)
    print(f"Moving averages with window_size={window_size}:")
    print(moving_averages)
    print("Graph of moving averages of survival rates.")
    # This range is because survival rates cannot be computed until the second year.
    # Then the moving average cannot be calculated until the <window_size> has elapsed.
    # This results in the first data point occurring in year <1+window_size>.
    plt.plot(
        range(1 + window_size, len(moving_averages) + window_size + 1), moving_averages
    )
    plt.xlabel("time")
    plt.ylabel(f"average rate, window {window_size}")
    plt.title("moving averages of survival_rates")
    plt.show()

    # MOVING AVERAGES LINEAR FIT.
    x = range(1 + window_size, len(moving_averages) + 1 + window_size)
    y = moving_averages
    coefficients = np.polyfit(x, y, 1)
    print("Linear Fit Coefficients:", coefficients)
    # Create polynomial function
    p = np.poly1d(coefficients)
    plt.scatter(x, y, label="Data Points")
    plt.plot(x, p(x), label="Linear Fit", color="red")
    plt.legend()
    plt.title("Linear fit of moving averages")
    plt.show()

    # Graph moving averages+ linear fit, increasing number of points each iteration.
    print("Graphs of moving averages with a linear fit.")
    coefficients_list = []
    for i in range(1, len(moving_averages) + 1):
        x = range(1 + window_size, i + 1 + window_size)
        y = moving_averages[:i]
        coefficients = np.polyfit(x, y, 1)
        p = np.poly1d(coefficients)
        plt.scatter(x, y, label="Data Points")
        plt.plot(x, p(x), label="Linear Fit", color="red")
        ax = plt.gca()
        ax.set_ylim((0.4, 0.6))
        plt.title("moving_averages with linear fit")
        plt.legend()
        coefficients_list.append(coefficients[0])
        print(f"{coefficients_list[-1]:.20f}")
        plt.show()

    print("Graph of coefficients from moving averages of survival rates.")
    # Coefficients graph from the survival rates, this is the value that would be given in real time.
    x = range(1 + window_size, len(coefficients_list) + 1 + window_size)
    y = coefficients_list
    plt.plot(x, y)
    plt.title("coefficients from moving averages of survival rates")
    plt.show()

    percent_changes = [
        round(i * 100, 4) for i in calculate_percent_change(survival_rates)
    ]
    print("percent changes=", percent_changes)

    # plt.plot(
    #     range(1, len(percent_changes) + 1, 1),
    #     percent_changes,
    #     marker="o",
    #     linestyle="-",
    #     color="b",
    # )
    # plt.xlabel("year")
    # plt.ylabel("percent change")
    # plt.title("")
    # # plt.legend()
    # plt.show()

    # percent_changes = [
    #     round(i * 100, 4) for i in calculate_percent_change(survival_rates)
    # ]

    return


if __name__ == "__main__":
    args = parse_arguments()
    config(
        use_logger=args.use_logger,
        verbose=args.verbose,
    )
    logging.debug("----- BEGIN PROGRAM -----")
    main(args)
