import logging

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

    def germinate_seeds(self, rate) -> None:
        """Seeds germinate from the seedbank at the given rate.

        Seeds are transferred from the underground population to the
        overground population. Rate should be a value [0-1] where 1
        means all of the seeds are transferred.
        """

        # TODO ? Add stochastic rate.
        logger.debug("Germinating seeds...")
        if rate < 0 or rate == 0 or rate > 1:
            logger.warning(f"germinate_seeds has a unexpected rate: {rate}")

        seed_pop = self._get_population_object(location="underground")
        pop_dict = seed_pop.get_population()
        seeds_dict = {key: pop_dict[key] * rate for key in pop_dict.keys()}

        logger.debug(f"Overground...")
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

        logger.debug(f"Returning seeds to seedbank...")
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

    def _get_population_object(self, location):
        assert location in [
            "underground",
            "overground",
        ], "location should be ['underground','overground']"

        if location == "underground":
            return self.seed_pop_underground
        elif location == "overground":
            return self.seed_pop_overground
        else:
            logging.error("location should be ['underground','overground']")

    def purge_population(
        self, amount_to_remove, chromosome_list, location, print_to_console=True
    ) -> None:
        """Remove an absolute value or percentage from the seed
        population.

        The amount to remove is assumed to be a percentage if <=1. This
        action is printed to the console for clarity to the user. If
        this is not desirable use print_to_console=False or
        remove_seeds() for silent usage. chromosome_list is a list of
        chromosomes to remove, e.g. ["rr"] or ["rR", "Rr", "rr"].
        """

        if amount_to_remove < 0:
            logging.warning("Attempted population purge with negative value, ignoring.")
            return

        seed_pop = self._get_population_object(location)

        # Assume percentage.
        if amount_to_remove <= 1:
            if not chromosome_list:
                chromosome_list = seed_pop.get_population().keys()
            for chromosome in chromosome_list:
                if print_to_console:
                    print(
                        f"Purging {amount_to_remove*100}% of {chromosome} seeds from"
                        f" {location}."
                    )
                count = seed_pop.get_population()[chromosome]
                seed_pop.remove_seeds(chromosome, count * amount_to_remove)

            return

        # Otherwise assume absolute value.
        for chromosome in chromosome_list:
            if print_to_console:
                print(f"Purging {amount_to_remove} {chromosome} seeds from {location}.")
            seed_pop.remove_seeds(chromosome, count)

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

        for chromosome, count in new_counts.items():
            seed_pop.add_seeds(chromosome, count)
        logger.info(f"Calculated population change.")
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
