import logging
import random

import numpy as np

import population_model

logger = logging.getLogger(__name__)


# TODO add logging
class ObserverModel:

    def __init__(
        self,
        accuracy: float = 1.0,
        fpr: float = 0.0,
        noise_standard_dev: float = 5.0,
    ):
        """Create an observer. Use the given accuracy and std-dev for noise.

        fpr : float : false positive rate. 1.0 is 100% misclassification.
        """
        self.observation_accuracy = accuracy

        self.observation_noise_standard_dev = noise_standard_dev
        self.fpr = fpr
        logger.info(
            f"Accuracy set to {self.observation_accuracy},"
            f" noise set to {self.observation_noise_standard_dev}."
            " This means practical minimum is:"
            f" {round(self.observation_accuracy-3*self.observation_noise_standard_dev,2)}"
            " and practical maximum is"
            f" {round(self.observation_accuracy+3*self.observation_noise_standard_dev,2)}"
        )

    def observe(
        self,
        pop_model: population_model.PopulationModel,
        accuracy=None,
        noisy=False,
        other_populations: list = None,
        fpr=None,
    ):
        """Observe the overground population. Returns count of population.

        accuracy : float
            - By specificying an accuracy, the class variable
            observation_accuracy can be overrided.
        other_populations : list[int/float]
            - List of populations that will combine with the FPR of the
            observer.
        noisy : bool
            - Compute a noisy accuracy or not.
            - self.noise_standard_dev class variable controls the
            variability of this noise.
        """

        if accuracy is None:
            accuracy = self.observation_accuracy

        if not (0 <= accuracy <= 1):
            logger.error(
                "Observation accuracy may be set incorrectly:",
                accuracy,
            )

        if fpr is None:
            fpr = self.fpr

        if not (0 <= fpr <= 1):
            logger.error("FPR may be set incorrectly:", fpr)

        POPULATION = pop_model.get_population(location="overground")
        count = sum(POPULATION.values())
        # poisson = np.random.poisson(count, 1)
        other_populations.insert(0, count)
        # Randomly 'observe' the populations.
        noisy_all_populations = (
            [
                np.random.binomial(pop, self.observation_accuracy)
                for pop in other_populations
            ]
            if noisy
            else other_populations
        )

        num_individuals_fp_from_each_pop = [i * fpr for i in noisy_all_populations]
        fpr_population = self._redistribute_counts(num_individuals_fp_from_each_pop)

        observed_count = noisy_all_populations[0] + fpr_population[0]
        return observed_count

        # if noisy:
        # accuracy = self.get_noisy_accuracy(
        # accuracy, self.observation_noise_standard_dev
        # )

        """count = 0
        POPULATION = pop_model.get_population(location="overground")
        for key, val in POPULATION.items():
            # Accuracy is returned as a list, but only has one value.
            count += val * accuracy
        """
        return count

    def _redistribute_counts(self, old_list) -> list:
        list_length = len(old_list)
        new_list = [0] * list_length
        for index, val in enumerate(old_list):
            numbers_to_add = self._random_split_number(val, list_length - 1)
            # Insert a 0 at index to make the list the same length.
            numbers_to_add.insert(index, 0)
            new_list = [sum(x) for x in zip(new_list, numbers_to_add)]
        return new_list

    def _random_split_number(self, total, list_length):
        # Generate n-1 random split points.
        split_points = sorted(random.randint(0, total) for _ in range(list_length - 1))
        # Add 0 and total to the split points to define the ranges.
        split_points = [0] + split_points + [total]
        # Calculate the differences between consecutive split points.
        return [split_points[i + 1] - split_points[i] for i in range(list_length)]

    # Unused?
    def get_noisy_accuracy(self, accuracy, noise_standard_dev) -> float:
        """TODO

        TODO
        """
        # TODO change from gaussian to a different distribution (idk which)
        noise = np.random.normal(accuracy, noise_standard_dev, 1)
        noise_list = list(map(float, noise))
        return noise_list[0]
