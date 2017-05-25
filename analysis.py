import numpy as np


def calculate_likelihood_diff(b_expected_outcomes, expected_outcomes):
    diff = np.subtract(expected_outcomes, b_expected_outcomes)
    total_diff = np.sum(np.absolute(diff))
    avg_diff = total_diff / len(expected_outcomes)
    return diff, total_diff, avg_diff


def calculate_entropy(expected_outcomes):
    entropy = np.zeros(len(expected_outcomes))
    for i, prob in enumerate(expected_outcomes):
        if prob != 0 and prob != 1:
            entropy[i] = (-prob * np.log2(prob)) + (-(1-prob) * np.log2(1-prob))
    return np.sum(entropy)/len(expected_outcomes)