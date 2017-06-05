import numpy as np
import math


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
    np.seterr(all='warn')
    entropy = np.zeros(len(expected_outcomes))
    for i, prob in enumerate(expected_outcomes):
        if 0 < prob < 1:
            entropy[i] = (-prob * np.log2(prob)) + (-(1-prob) * np.log2(1-prob))
        elif math.isnan(prob):
            print("# # # # # # # Prob is ", prob)
    return np.sum(entropy)/len(expected_outcomes)


def calculate_agreement(policy, profile, type=2):
    """
    Returns a vector of voter agreement calculated through a specified method
    :param policy: the implemented policy
    :param profile: the profile of the electorate
    :param type: the calculation method. 1: manhattan distance, 2: absolute difference, 3: bayesian agreement
    :return:  a vector of voter agreement calculated through a specified method
    """
    agreements = np.zeros(len(profile))
    I = len(policy)

    # Case 1: Mahattan distance
    if type == 1:
        for i, voter in enumerate(profile):
            agreements[i] = (I - np.sum(np.absolute(np.subtract(voter, policy))))/I

    # Case 2: Cosine similarity
    if type == 2:
        for i, voter in enumerate(profile):
            disagreements = np.sum(np.absolute(np.subtract(voter, policy)))
            agreements[i] = (I-2*disagreements) / I

    # Case 3: Bayesian agreement
    if type == 3:
        # TODO
        pass

    return agreements