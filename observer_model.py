import itertools
import logging
import random
from copy import deepcopy

import numpy as np

import population_model

logger = logging.getLogger(__name__)


# TODO add logging
class ObserverModel:

    def __init__(self, tpr: float = 1.0):
        """Create an observer. Use the given TPR (true positive rate).

        The false negative rate will be calculated using the given TPR
            (FPR = 1 - TPR).
        Other noise parameters are defined within the observe function.
        tpr (true positive rate) : float : [0-1]
        """
        self.set_tpr_and_fnr(tpr)

    def calculate_fnr(self, tpr) -> float:
        """Calculate false negative rate. Return as a float."""
        fnr = 1 - tpr
        return fnr

    def set_tpr_and_fnr(self, tpr) -> None:
        self.tpr = tpr
        self.fnr = self.calculate_fnr(self.tpr)
        logger.info(f"TPR  set to {self.tpr}," f" FNR set to {self.fnr}.")
        return

    def observe(
        self,
        pop_model: population_model.PopulationModel,
        tpr=None,
        noisy=False,
        other_populations: list = [],
    ) -> float:
        """Observe the overground population. Returns count of population.

        pop_model : PopulationModel class instance.
        tpr : float
            - By specificying a true positive rate (TPR), the class
            variable tpr can be overridden.
        other_populations : list[int/float]
            - List of populations that will combine with the TPR/FNR of
            the observer.
        noisy : bool
            - Compute a noisy accuracy or not.
            - self.noise_standard_dev class variable controls the
            variability of this noise. TODO depreciated.
        NOTE: False negative rate (fnr) will be calculated with the
            given TPR value. If none is given the class FNR
            (calculated using the class TPR) will be used.
        """

        if tpr is None:
            tpr = self.tpr
            fnr = self.fnr
            logger.debug(
                f"No given TPR, using class TPR {self.tpr},"
                f" using class FNR {self.fnr}."
            )

        if not (0 <= tpr <= 1):
            logger.error("Observation TPR may be set incorrectly:", tpr)

        if not (0 <= fnr <= 1):
            logger.error("Observation FNR  may be set incorrectly:", fnr)

        if fnr > 0 and len(other_populations) == 0:
            logger.error("FPR set > 0 but no other_population list was given.")

        # Add the overground population to the other_population list.
        OVERGROUND_POPULATION = pop_model.get_population(location="overground")
        overground_count = sum(OVERGROUND_POPULATION.values())

        # Randomly 'observe' the populations using a multinomial
        # distribution. Result of [# TPs, # FNs] sums to 1 (no noise).
        if noisy:
            multinomials = [
                # np.random.binomial(pop, tpr)
                np.random.multinomial(pop, [tpr, fnr])
                for pop in itertools.chain([overground_count], other_populations)
            ]
            # The false negatives are randomly distributed to the other
            # populations.
            all_populations = self._redistribute_multinomials(multinomials)
            # Map np.float to float.
            all_populations = list(map(float, all_populations))
        else:
            # Copy the list so we do not have two variables for the same list.
            all_populations = deepcopy(other_populations).insert(0, overground_count)
        logger.debug(f"Noisy set to {noisy}. all_populations=")
        logger.debug(all_populations)

        observed_count = all_populations[0]
        logger.debug(f"observe: returning observed_count={observed_count}")
        return observed_count

    def _redistribute_multinomials(self, multinomials_list) -> list:
        """
        From a list of multinomials [[TP,FN],[TP,FN],...], distribute
        the FNs between the other indexes.
        e.g. [[9,1],[20,5],[30,10]]:
            1. The 1 gets split into [0, 0.47..., 0.52...].
                - Random split, add 0 to INDEX (0 in this case).
            2. The 5 gets split into [4.51..., 0, 0.48...].
                - Index is now 1..., etc.
            3. The 10 gets split into [8.09..., 1.90..., 0].
            4. The TP values are made into a new list: [9, 20, 30].
            5. All lists are added together: [21.6..., 22.37..., 31...].
        """
        list_length = len(multinomials_list)
        new_list = [0] * list_length

        for index, val in enumerate(multinomials_list):
            tp, fn = val
            new_list[index] += tp
            # Split the false negative number into a list of smaller numbers.
            # e.g. 10 -> [3,7]. Then insert 0 at index, e.g. [0,3,7] to
            # allow this list and new_list to be added together.
            false_negatives_to_add = self._random_split_number(fn, list_length - 1)
            false_negatives_to_add.insert(index, 0)
            logger.debug(
                f"_redistribute_multinomials: false_negatives_to_add={false_negatives_to_add}"
            )
            new_list = [sum(x) for x in zip(new_list, false_negatives_to_add)]
        logger.debug(f"_redustribute_multinomials: new_list={new_list}")
        return new_list

    # TODO unused, remove.
    def _redistribute_counts(self, old_list) -> list:
        """From a list of numbers, randomly distribute each index into every other index in the list.

        e.g. [10,0,0] -> [0,4,6]
        e.g. [10,5,0] -> The 10 becomes [0,4,6] and the 5 becomes [2,0,3] -> [2,4,9].
        The index the number originates from cannot contain any number from the original count.
        Returns list of new (redistributed) numbers.
        """
        logger.warning(
            "_redistribute_counts() is depreciated. Use _redistribute_multinomials() instead."
        )
        list_length = len(old_list)
        new_list = [0] * list_length
        for index, val in enumerate(old_list):
            numbers_to_add = self._random_split_number(val, list_length - 1)
            # Insert a 0 at index to make the list the same length.
            numbers_to_add.insert(index, 0)
            logger.debug(f"numbers_to_add: {numbers_to_add}")
            new_list = [sum(x) for x in zip(new_list, numbers_to_add)]
        logger.debug(f"_redistribute_counts(): {new_list}")
        return new_list

    def _random_split_number(self, total, list_length):
        # Generate n-1 random split points.
        split_points = sorted(random.uniform(0, total) for _ in range(list_length - 1))
        # Add 0 and total to the split points to define the ranges.
        split_points = [0] + split_points + [total]
        # Calculate the differences between consecutive split points.
        nums = [split_points[i + 1] - split_points[i] for i in range(list_length)]
        # logger.debug(f"_random_split_number: {nums}")
        return nums
