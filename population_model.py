import logging
from copy import copy

import numpy as np

import seed_population

logger = logging.getLogger(__name__)


class PopulationModel:
    """Gives structure and helper functions to the SeedPopulation class.


    Attributes:
        seed_pop_underground : SeedPopulation
        seed_pop_overground : SeedPopulation

    Methods:
        germinate_seeds(rate)
            Transfers seeds from the underground population to the
            overground population.
        return_seeds_to_seedbank(rate=1.0)
            Transfers seeds from the overground population to the
            underground population. This rate will often be 1.0.
        add_seeds(chromosome, count, location)
            Add {count} {chromosome} seeds to {location} ['underground','overground']
        remove_seeds(chromosome, count, location)
            Remove {count} {chromosome} seeds from {location} ['underground','overground']
        replace_seeds(chromosome, count, location)
            Replace {count} {chromosome} seeds at {location} ['underground','overground']
        purge_population
        apply_population_change -> dict
        get_population -> dict
        get_frequency -> dict
        _get_population_object -> SeedPopulation (one of the seed_pop attributes)
        _get_chromosome_pairing_output -> list
        _split_chromosome -> list
    """

    def __init__(self, chromosome_dict={"RR", "Rr", "rR", "rr"}):
        """Creates the SeedPopulation object with default chromosomes.

        default chromosomes = 'RR','Rr','rR','rr'.
        """

        self.seed_pop_underground = seed_population.SeedPopulation(
            chromosome_dict=chromosome_dict
        )
        self.seed_pop_overground = seed_population.SeedPopulation(
            chromosome_dict=chromosome_dict
        )
        logger.debug("Created PopulationModel instance.")

    def germinate_seeds(self, rate, noisy=False) -> None:
        """Seeds germinate from the seedbank at the given rate.

        Seeds are transferred from the underground population to the
        overground population. Rate should be a value [0-1] where 1
        means all of the seeds are transferred.
        """

        logger.debug("Germinating seeds...")
        if rate < 0 or rate == 0 or rate > 1:
            logger.warning(f"germinate_seeds has a unexpected rate: {rate}")

        underground_population = self._get_population_object(location="underground")
        seeds_dict = underground_population.get_population()

        # Each chromosome is selected from binomial distribution.
        if noisy:
            for chromosome, count in seeds_dict.items():
                seeds_dict[chromosome] = np.random.binomial(count, rate)
        else:
            for chromosome, count in seeds_dict.items():
                seeds_dict[chromosome] = count * rate

        logger.debug("Overground...")
        seed_pop = self._get_population_object(location="overground")
        for chromosome, count in seeds_dict.items():
            seed_pop.add_seeds(chromosome, count)

        logger.debug("Underground...")
        seed_pop = self._get_population_object(location="underground")
        for chromosome, count in seeds_dict.items():
            seed_pop.remove_seeds(chromosome, count)

        logger.info(f"Germinated seeds at rate {rate}.")
        return

    def return_seeds_to_seedbank(self, rate=1.0) -> None:
        """Seeds from the overground population get returned to the
        underground population.
        """

        logger.debug("Returning seeds to seedbank...")
        if rate < 0 or rate == 0 or rate > 1:
            logger.warning(f"return_seeds_to_seedbank has a unexpected rate: {rate}")

        seed_pop = self._get_population_object(location="overground")
        pop_dict = seed_pop.get_population()
        seeds_dict = {key: pop_dict[key] * rate for key in pop_dict.keys()}

        logger.debug("Underground...")
        for chromosome, count in seeds_dict.items():
            self.seed_pop_underground.add_seeds(chromosome, count)

        logger.debug("Overground...")
        for chromosome, count in seeds_dict.items():
            self.seed_pop_overground.remove_seeds(chromosome, count)
        logger.info(f"Returned seeds to seedbank at rate {rate}.")
        return

    def add_seeds(self, chromosome, count, location) -> None:
        """Adds the given number of seeds to the overground population."""
        logger.info(f"Adding seeds to {location}.")
        seed_pop = self._get_population_object(location)
        seed_pop.add_seeds(chromosome, count)
        return

    def remove_seeds(self, chromosome, count, location) -> None:
        """Removes the given number of seeds from the population.

        Passing -1 as count will remove all seeds of a type from the
        population.
        """

        logger.info(f"Removing seeds from {location}.")
        seed_pop = self._get_population_object(location)

        if count == -1:
            population = seed_pop.get_population(key=chromosome)[chromosome]
            count = population

        seed_pop.remove_seeds(chromosome, count)
        return

    def _get_population_object(self, location) -> seed_population.SeedPopulation:
        assert location in [
            "underground",
            "overground",
        ], "location should be ['underground','overground']"

        if location == "underground":
            return self.seed_pop_underground
        elif location == "overground":
            return self.seed_pop_overground

    def purge_population(
        self,
        amount_to_remove,
        location,
        chromosome_list=None,
        noisy=False,
        # print_to_console=True,
    ) -> None:
        """Remove an absolute value or percentage from the seed
        population.

        The amount to remove is assumed to be a percentage if <=1.
        location : ['overground, 'underground'] - typically overground.
        chromosome_list is a list of chromosomes to remove,
            e.g. ["rr"] or ["rR", "Rr", "rr"].
        The noisy flag enables sampling from a:
            - Poisson distribution (for absolute values),
            - Binomial distribution (for percentages).
        """

        if amount_to_remove < 0:
            logger.warning("Attempted population purge with negative value, ignoring.")
            return

        if chromosome_list is not None and len(chromosome_list) == 0:
            logger.warning("purge_population: chromosome_list is empty.")

        seed_pop = self._get_population_object(location)
        current_population = seed_pop.get_population()

        if not chromosome_list:
            chromosome_list = seed_pop.get_population().keys()

        rng = np.random.default_rng()
        for chromosome in chromosome_list:
            if amount_to_remove <= 1:  # Assume percentage.
                logger.info(
                    f"Purging {amount_to_remove*100}%"
                    f" of population ({chromosome})"
                    f" from {location}..."
                )
                if noisy:
                    absolute_to_remove = rng.binomial(
                        current_population[chromosome], amount_to_remove
                    )
                else:
                    absolute_to_remove = (
                        current_population[chromosome] * amount_to_remove
                    )

            else:  # Assume absolute value.
                logger.info(
                    f"Purging {absolute_to_remove} {chromosome} seeds from {location}."
                )
                if noisy:
                    absolute_to_remove = rng.poisson(amount_to_remove)
                else:
                    absolute_to_remove = copy(amount_to_remove)

            seed_pop.remove_seeds(chromosome, absolute_to_remove)

        return

    def apply_population_change(self) -> dict:
        """Calculate resultant children from two parents' chromosomes.

        Calculates the offspring for each chromosome pairing.
        """

        logger.debug("Calculating population change...")
        seed_pop = self._get_population_object(location="overground")
        new_counts = seed_pop.get_blank_count_dict()

        # Calculate frequency of each seed type within the parent population.
        frequencies = seed_pop.get_frequency()

        # Calculate the proportions of new seeds for each gene combination.
        for chromosome1, frequency1 in frequencies.items():
            for chromosome2, frequency2 in frequencies.items():
                children = self._get_chromosome_pairing_output(chromosome1, chromosome2)
                logger.debug(f"{chromosome1}, {chromosome2} children: {children}")
                # Division by 0.25 as there is there are four options
                # for the children (e.g. RR, Rr, rR, and rr).
                new_seed = (
                    sum(seed_pop.get_population().values())
                    * frequency1
                    * frequency2
                    * 0.25
                )

                logger.debug(
                    f"{sum(seed_pop.get_population().values())}"
                    f" * {frequency1:.3f} * {frequency2:.3f} * 0.25"
                    f" = {new_seed} seeds/child"
                )
                for child in children:
                    new_counts[child] += new_seed

        # The new seeds get deposited into the underground population.
        seed_pop = self._get_population_object(location="underground")
        for chromosome, count in new_counts.items():
            seed_pop.add_seeds(chromosome, count)
        logger.info("Calculated population change.")
        return new_counts

    def get_population(self, location) -> dict:
        """Returns the real count of each chromosome within the
        population.

        e.g. {'rR': 3.75, 'rr': 21.25, 'RR': 1.25, 'Rr': 13.75}
        """

        seed_pop = self._get_population_object(location)
        population = seed_pop.get_population()
        logger.debug(f"Population: {population}")
        return population

    def get_frequency(self, location) -> dict:
        """Returns the frequency of each chromosome within the
        population.

        location : ['underground','overground']
        n_decimals : int : number of decimals to round dict values to.
        e.g. {'rR': 0.09375, 'rr': 0.53125, 'RR': 0.03125,
        'Rr': 0.34375}
        """

        seed_pop = self._get_population_object(location)
        freq = seed_pop.get_frequency()
        logger.debug(f"Frequency: {freq}")
        return freq

    def _get_chromosome_pairing_output(self, chromosome1, chromosome2) -> list:
        """Gets all permutations of two chromosomes e.g.
        Rr+Rr->['RR','Rr','rR','rr'].

        Whilst this function returns permutations (e.g. Rr and rR are
        different chromosomes, this is not the case in genetics where rR
        is usually also shown as Rr. This process in genetics is
        actually combinations (order does not matter).
        """

        if len(chromosome1) != 2 or len(chromosome2) != 2:
            print("Error: chromosome wrong length:", chromosome1, chromosome2)

        output = []
        for gene1 in self._split_chromosome(chromosome1):
            for gene2 in self._split_chromosome(chromosome2):
                output.append(f"{gene1}{gene2}")
        return output

    def _split_chromosome(self, chromosome) -> list:
        """Split a chromosome into its parts e.g. XY becomes ['X','Y'].

        This is a separate function to allow for expansion later.
        """

        genes = []
        for gene in chromosome:
            genes.append(gene)
        return genes
