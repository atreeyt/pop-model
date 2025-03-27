import logging
from copy import copy

logger = logging.getLogger(__name__)


class SeedPopulation:
    """Seed population is stored within a dictionary.

    The seeds are modelled as a simple one chromosome plant (haploidy).
    This is expressed as a dominant resistant (R) gene and a recessive
    susceptible (r) gene.

    Attributes:
        seed_counts : dict
            Stores the chromosome types of the population and their
            respective counts.
    """

    def __init__(self, chromosome_dict={"RR", "Rr", "rR", "rr"}):
        """Create dictionary for the counts of each seed type."""

        self.seed_counts = {}
        for key in chromosome_dict:
            self.seed_counts[key] = 0
        logger.debug("SeedPopulation instance created.")

    def get_population(self, key=None) -> dict:
        """Return the population statistics. Returns dict.

        e.g. {"RR": 0, "Rr": 30, "rR": 30, "rr": 1000}.
        """
        if key:
            return {key: self.seed_counts[key]}
        return copy(self.seed_counts)

    def get_frequency(self, find_key=None) -> dict:
        """Returns the frequency of each seed type within the population.

        Returns population of find_key.
        If find_key is None then returns all frequencies.
        """

        total_seeds = sum(self.seed_counts.values())
        # Guard against division by 0.
        if total_seeds == 0:
            logger.debug("get_frequency: seed_count is empty, returning zeroes.")
            return dict.fromkeys(self.seed_counts.keys(), 0.0)

        freq = {}
        for key, val in self.seed_counts.items():
            freq[key] = val / total_seeds

        if find_key:
            logger.debug(f"get_frequency (key={find_key}):", freq[find_key])
            return freq[find_key]
        logger.debug(f"get_frequency: {freq}")
        return freq

    def add_seeds(self, chromosome, count) -> None:
        """Increase value in seed_counts dict of chromosome key by count.

        Warning given for adding seeds of an unknown chromosome.
        """

        if chromosome not in self.seed_counts.keys():
            logger.warning(
                f"Seed type {chromosome} does not exist in population already."
                f" Adding {count:_} seeds. Is this expected?"
            )
            self.seed_counts[chromosome] = 0
        self.seed_counts[chromosome] += count
        logger.debug(f"add_seeds: {chromosome} {count:_}")
        return

    def remove_seeds(self, chromosome, count) -> None:
        """Set value to 0 in seed_counts dict for chromosome key.

        Warnings are given for unknown an chromosome or for removing
        more seeds than are currently in the population.
        """

        if chromosome not in self.seed_counts.keys():
            logger.warning(
                f"Attempted to remove {count:_} {chromosome} seeds from population but"
                " this seed type does not exist in the population. No effect."
            )
            return

        current_count = self.seed_counts[chromosome]
        if current_count < count:
            logger.warning(
                f"Attempted to remove {count:_} {chromosome} seeds from population when"
                f" only {current_count} seeds exist. Setting to 0."
            )
            self.seed_counts[chromosome] = 0
        else:
            self.seed_counts[chromosome] -= count
        logger.debug(f"remove_seeds: {chromosome} {count:_}")
        return

    def replace_seeds(self, chromosome, count):
        """Replace the value in seed_counts dict with new count for chromosome key.

        A warning is given for attempting to replace seeds of an unknown
        chromosome. In this case the seeds are ADDED to the population.
        """

        if chromosome not in self.seed_counts.keys():
            logger.warning(
                f"Attempted to replace {count:_} {chromosome} seeds from population but"
                " this seed type does not exist in the population. ADDING seeds of new"
                " type. Is this expected behaviour?"
            )
        self.seed_counts[chromosome] = count
        return

    def get_blank_count_dict(self) -> dict:
        """Return a dict with the same keys as seed_counts with each value set to 0."""

        blank_dict = {}
        for key in self.seed_counts.keys():
            blank_dict[key] = 0
        logger.debug("Retrieved blank count dictionary.")
        return blank_dict
