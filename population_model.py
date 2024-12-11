import logging

import seed_population
import utils

logger = logging.getLogger(__name__)


class PopulationModel():
    def __init__(self, chromosome_dict={'RR','Rr','rR','rr'}):
        """Creates the SeedPopulation object with default chromosomes.
        
        default chromosomes = 'RR','Rr','rR','rr'.
        """
        self.seed_pop = seed_population.SeedPopulation(chromosome_dict=chromosome_dict)
        logger.debug('Created PopulationModel instance.')

    def add_seeds(self, chromosome, count) -> None:
        """Adds the given number of seeds to the population."""
        logger.info(f'add_seeds: {chromosome} {count}')
        count_dict = {chromosome: count}
        self.seed_pop.update_counts(count_dict, replace=False,
                                    remove_others=False)
        return

    def remove_seeds(self, chromosome, count) -> None:
        """Removes the given number of seeds from the population."""
        population = self.seed_pop.get_population(key=chromosome)[chromosome]
        # Case for removing all seeds of a type from the population.
        if count == -1:
            count = population
        if count > population:
            logger.warning(f'Attempted to remove more seeds than currently in'
                           f' the population: {chromosome},'
                           f' current: {population}, to remove: {count}.'
                           ' Removed whole population.')
            count = population
        # Add a negative value (remove).
        self.add_seeds(chromosome, -count)
        return
    
    def replace_seeds(self, count_dict, remove_others) -> None:
        """Updates the seed population object with the new seed counts.
        
        The seed population object has other seed values REMOVED and the new
        ones entered in its place, when remove_others=True.
        """
        logger.info(f'replace_seeds: {count_dict}')
        self.seed_pop.update_counts(count_dict, remove_others=remove_others)
        return
    
    def calculate_population_change(self) -> dict:
        """Calculate resultant children from two parents' chromosomes.

        Calculates the offspring for each chromosome pairing.
        """
        logger.debug('Calculating population change...')
        new_counts = self.seed_pop.get_blank_count_dict()
        # Calculate frequency of each seed type within the parent population.
        frequencies = self.seed_pop.get_frequency()
        
        # Calculate the proportions of new seeds for each gene combination.
        for chromosome1, frequency1 in (frequencies.items()):
            for chromosome2, frequency2 in (frequencies.items()):
                children = self._get_chromosome_pairing_output(chromosome1,
                                                              chromosome2)
                logger.debug(f'{chromosome1}, {chromosome2} children: {children}')
                # Division by 0.25 as there is there are four options for
                # the children (e.g. RR, Rr, rR, and rr).
                new_seed = sum(self.seed_pop.get_population().values()) \
                        * frequency1 \
                        * frequency2 \
                        * 0.25
                # TODO remove logger below, replace with something more helpful/formatted better
                logger.debug(
                    f'{sum(self.seed_pop.get_population().values())}'
                    f' * {frequency1:.3f} * {frequency2:.3f} * 0.25'
                    f' = {new_seed} seeds/child'
                )
                for child in children:
                    new_counts[child] += new_seed
                    
        logger.info(f'Calculated population change.')
        self.replace_seeds(new_counts, remove_others=False)
        return new_counts

    def get_population(self) -> dict:
        """Returns the real count of each chromosome within the population.
        
        e.g. {'rR': 3.75, 'rr': 21.25, 'RR': 1.25, 'Rr': 13.75}
        """
        pop = self.seed_pop.get_population()
        logger.info(f'Population: {utils.round_dict_values(pop)}')
        return pop
    
    def get_frequency(self) -> dict:
        """Returns the frequency of each chromosome within the population.
        
        e.g. {'rR': 0.09375, 'rr': 0.53125, 'RR': 0.03125, 'Rr': 0.34375}
        """
        freq = self.seed_pop.get_frequency()
        logger.info(f'Frequency: {utils.round_dict_values(freq)}')
        return freq

    def _get_chromosome_pairing_output(self, chromosome1, chromosome2) -> list:
        """Gets all combinations/permutations (idk which) of two chromosomes."""
        # TODO know which ^^
        if len(chromosome1) != 2 or len(chromosome2) != 2:
            print('Error: chromosome wrong length:', chromosome1, chromosome2)

        output = []
        for gene1 in self._split_chromosome(chromosome1):
            for gene2 in self._split_chromosome(chromosome2):
                output.append(f'{gene1}{gene2}')
        return output

    def _split_chromosome(self, chromosome) -> list:
        """Split a chromosome into its parts e.g. XY becomes ['X','Y'].

        This is split to allow for expansion later.
        """
        genes = []
        for gene in chromosome:
            genes.append(gene)
        return genes
