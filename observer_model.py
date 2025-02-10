import logging

import numpy as np

import population_model

logger = logging.getLogger(__name__)


# TODO add logging
class ObserverModel:

    def __init__(
        self,
        observation_accuracy: float = 1.0,
        noise_standard_dev: float = 5.0,
        other_populations=None,
        fpr=0.0,
    ):
        """Create an observer. Use the given accuracy and std-dev for noise.

        other_populations : list[int/float] : List of populations that will combine with the FPR of the observer.
        fpr : float : false positive rate. 1.0 is 100% misclassification.
        """
        self.observation_accuracy = observation_accuracy
        self.observation_noise_standard_dev = noise_standard_dev
        self.other_populations = other_populations
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
        other_populations=None,
        fpr=None,
    ):
        """Observe the overground population. Returns count of population.

        By specificying an accuracy, the class variable observation_accuracy
        can be overrided.

        noisy : bool : Compute a noisy accuracy or not.
        self.noise_standard_dev : Class variable controls the
            variability of this noise.
        """

        # TODO when observing, the observer's FP rate will change the
        # accuracy of observations. Some observations will be
        # incorrectly classified as another species. The population of
        # these species matters HIGHLY. A 1% FP rate will mean that with
        # a population of 1000, 10 species will be incorrectly
        # classified as another species.
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

        if other_populations is None:
            other_populations = self.other_populations
            # TODO implement fpr inaccuracies

        # TODO ? Should this be in loop below? If yes, accuracy will
        # be noisy per chromosome. Probably overkill.
        if noisy:
            accuracy = self.get_noisy_accuracy(
                accuracy, self.observation_noise_standard_dev
            )

        count = 0
        POPULATION = pop_model.get_population(location="overground")
        for key, val in POPULATION.items():
            # Accuracy is returned as a list, but only has one value.
            count += val * accuracy

        return count

    def get_noisy_accuracy(self, accuracy, noise_standard_dev) -> float:
        """TODO

        TODO
        """
        # TODO change from gaussian to a different distribution (idk which)
        noise = np.random.normal(accuracy, noise_standard_dev, 1)
        noise_list = list(map(float, noise))
        return noise_list[0]
