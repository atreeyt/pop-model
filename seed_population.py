import logging

logger = logging.getLogger(__name__)


class SeedPopulation:
    """Seed population is stored within a dictionary.

    The seeds are modelled as a simple one chromosome plant (haploidy).
    This is expressed as a dominant resistant (R) gene and a recessive
    susceptible (r) gene.
    """
    def __init__(self, chromosome_dict={'RR','Rr','rR','rr'}):
        """Create dictionary for the counts of each seed type."""
        # Initialise all counts to zero.
        self.seed_counts = {}
        for key in chromosome_dict:
            self.seed_counts[key] = 0
        logger.debug('SeedPopulation instance created.')

    def get_population(self, key=None) -> dict:
        """Return the population statistics."""
        if key:
            return { key: self.seed_counts[key] }
        return self.seed_counts
    
    def get_frequency(self, find_key=None) -> dict:
        """Returns the frequency of each seed type within the population.

        Returns population of find_key.
        If find_key is None then returns all frequencies.
        """
        total_seeds = sum(self.seed_counts.values())
        # Guard against division by 0.
        if total_seeds == 0:
            logger.debug('get_frequency: seed_count is empty, returning zeroes.')
            return dict.fromkeys(self.seed_counts.keys(), 0.0)
        
        freq = {}
        for key, val in self.seed_counts.items():
            freq[key] = val / total_seeds

        if find_key:
            logger.debug(f'get_frequency (key={find_key}):', freq[find_key])
            return freq[find_key]
        logger.debug(f'get_frequency: {freq}')
        return freq
    
    def update_counts(self, seed_count_dict, replace=False,
                      remove_others=False) -> None:
        """Replaces/updates the values for the seed count.

        If replace=True then the seed count is replaced, otherwise it is added.
            This addition can be negative.
        If remove_others=True then ALL other seed values not within
            seed_count_dict are removed.
        """
        if remove_others:
            # Override all counts and replace dictionary.
            self.seed_counts = {}
        # Only replace given key-val pairs.
        for key, val in seed_count_dict.items():
            if replace or remove_others:
                self.seed_counts[key] = val
            else:
                self.seed_counts[key] += val
                if self.seed_counts[key] < 0:
                    logger.warning(f'{self.seed_counts[key]} population is <0, setting to 0.')
        logger.debug(f'Updated counts (remove_others={remove_others}):')
        logger.debug(f'    {seed_count_dict} to {self.seed_counts}')
        return
    
    def get_blank_count_dict(self) -> dict:
        """Creates a population dict with each value set to 0 but with keys."""
        blank_dict = {}
        for key in self.seed_counts.keys():
            blank_dict[key] = 0
        logger.debug('Retrieved blank count dictionary.')
        return blank_dict
