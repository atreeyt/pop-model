import logging

import population_model
import utils
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)


# TODO add logging
class ObserverModel:

    def __init__(self, observation_accuracy, noise_standard_dev=5):
        if not (0 <= observation_accuracy <= 1):
            logger.error(
                "Observer model accuracy may be set incorrectly:",
                observation_accuracy,
            )
        self.observation_accuracy = observation_accuracy
        self.observation_noise_standard_dev = noise_standard_dev
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
    ):
        """TODO

        By specificying an accuracy, the class variable observation_accuracy
        can be overrided.
        """
        if accuracy and not (0 <= accuracy <= 1):
            logger.error(
                "Observe accuracy may be set incorrectly:",
                accuracy,
            )

        if accuracy == None:
            accuracy = self.observation_accuracy

        if noisy:
            accuracy = self.get_noisy_accuracy(
                accuracy, self.observation_noise_standard_dev
            )
        population = deepcopy(pop_model.get_population(location="overground"))
        for key, val in population.items():
            # Accuracy is returned as a list, but only has one value.
            population[key] = val * accuracy[0]

        return population

    def get_noisy_accuracy(
        self,
        accuracy,
        noise_standard_dev,
        amount_to_return=1,
        dp_to_round_results=2,
    ) -> list[float]:
        """TODO

        TODO
        """
        # TODO change from gaussian to a different distribution (idk which)
        noise = np.random.normal(accuracy, noise_standard_dev, amount_to_return)
        # Numpy returns an array of Numpy float types, so we convert here to
        # a regular list with regular floats. Round if required.
        noise_list = list(map(float, noise))
        noise_list = [round(x, dp_to_round_results) for x in noise_list]
        return noise_list
