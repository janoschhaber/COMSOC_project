import numpy as np
from scipy import spatial


def calculate_likelihood_diff(b_expected_outcomes, expected_outcomes):
    """
    Calculates the difference in likelihood between two expected outcomes (change from 2 to 1)
    :param b_expected_outcomes: (new) expected outcome
    :param expected_outcomes: (old (expected outcome)
    :return: the difference in likelihood between two expected outcomes
    """
    diff = np.subtract(expected_outcomes, b_expected_outcomes)
    total_diff = np.sum(np.absolute(diff))
    avg_diff = total_diff / len(expected_outcomes)
    return diff, total_diff, avg_diff


def calculate_entropy(expected_outcomes):
    """
    Calculates the entropy of an expected outcome
    :param expected_outcomes: expected outcome
    :return: the entropy of an expected outcome
    """
    entropy = np.zeros(len(expected_outcomes))
    for i, prob in enumerate(expected_outcomes):
        if prob != 0 and prob != 1:
            entropy[i] = (-prob * np.log2(prob)) + (-(1-prob) * np.log2(1-prob))
    return np.sum(entropy)/len(expected_outcomes)


def calculate_regret(policy, profile, type=2):
    """
    Returns a vector of voter regret calculated through a specified method
    :param policy: the implemented policy
    :param profile: the profile of the electorate
    :param type: the calculation method. 1: manhattan distance, 2: absolute difference, 3: bayesian regret
    :return:  a vector of voter regret calculated through a specified method
    """
    regrets = np.zeros(len(profile))
    I = len(policy)

    # Case 1: Mahattan distance
    if type == 1:
        for i, voter in enumerate(profile):
            regrets[i] = np.sum(np.absolute(np.subtract(voter, policy)))/I

    # Case 2: Cosine similarity
    if type == 2:
        for i, voter in enumerate(profile):
            disagreements = np.sum(np.absolute(np.subtract(voter, policy)))
            regrets[i] = (I-2*disagreements) / I

    # Case 3: Bayesian regret
    if type == 3:
        # TODO
        pass

    return regrets