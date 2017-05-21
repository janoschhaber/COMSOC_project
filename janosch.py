import numpy as np
import random as rd
from scipy import spatial
from itertools import chain, combinations



def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def simulate_poll(X, N, type=1, weightings=None):
    assert type < 4, "Voting error: Unknown poll type %i" % type
    result = None

    # Case 1: Uniform distribution of polls
    if type == 1:
        u_votes = int(N / X)
        result = np.full(X, u_votes)
        leftover = N % X
        if leftover > 0:
            additional = rd.sample(range(X), leftover)
            for c in additional:
                result[c] += 1

    # Case 2: Random distribution of polls
    elif type == 2:
        remaining = N
        for c in range(X - 1):
            votes = rd.randint(0, remaining)
            result[c] = votes
            remaining -= votes
        result[X - 1] = remaining

    # Case 3: Weighted distribution as specified
    elif type == 3:
        assert len(weightings) == X, "Voting error: More weights than parties specified"
        assert np.sum(weightings) == 1, "Voting error: Weights do not sum to 1"
        for c, weight in enumerate(weightings):
            result[c] = int(weight * N)
        leftover = N - int(np.sum(result))
        if leftover > 0:
            additional = rd.sample(range(X), leftover)
            for c in additional:
                result[c] += 1

    return result / N


def generate_profile(I, N, type=1, weight=None):
    assert type < 3, "Profile generation error: Unknown generation type %i" % type
    profile = None

    # Case 1: Uniform distribution of preferences
    if type == 1:
        profile = [np.random.choice([0, 1], size=(I,), p=[0.5, 0.5]) for _ in range(N)]

    # Case 2: Weighted distribution of preferences
    elif type == 2:
        profile = [np.random.choice([0, 1], size=(I,), p=[weight, 1 - weight]) for _ in range(N)]

    return profile


def generate_breakingpoints(X, I, B, strict=True):
    assert B <= I, "Breaking point error: Specified more breaking points than issues"
    breaking_points = [None] * X

    for c in range(X):
        b_issues = rd.sample(range(I), B)
        if not strict:
            remove = rd.randint(0, B)
            if remove > 0:
                b_issues = b_issues[:remove]
        breaking_points[c] = b_issues

    return breaking_points


def generate_coalitions(poll_res, agendas, breaking_points):
    assert len(poll_res) == len(agendas)
    X = len(poll_res)
    coalitions = list()

    possible_coaltitions = powerset([c for c in range(X)])
    for coalition in possible_coaltitions:
        forced_issues = dict()
        consistent = True
        poll_representation = 0
        for c in coalition:
            if not consistent:
                break
            poll_representation += poll_res[c]
            for b in breaking_points[c]:
                if b in forced_issues and forced_issues[b] != agendas[c][b]:
                    consistent = False
                    break
                elif b not in forced_issues:
                    forced_issues[b] = agendas[c][b]
        if consistent and poll_representation > 0.5:
            coalitions.append(coalition)

    return coalitions


def simulate_outcomes(poll_res, coalitions, agendas, breaking_points):
    expected_outcomes = [None] * len(coalitions)

    for c_index, coalition in enumerate(coalitions):
        coalition_agendas = [np.multiply(agendas[c],poll_res[c]) for c in coalition]
        outcome = np.sum(coalition_agendas, axis=0)
        relative_weight = 0
        for c in coalition:
            relative_weight += poll_res[c]
        outcome = outcome / relative_weight

        for c in coalition:
            c_breaking_points = breaking_points[c]
            for b in c_breaking_points:
                outcome[b] = agendas[c][b]

        expected_outcomes[c_index] = outcome

    return expected_outcomes



def rate_coalitions(coalitions, agendas, expected_outcomes, breaking_points, type = 1):
    ratings = np.zeros(len(coalitions))

    for c_index, coalition in enumerate(coalitions):
        expected_outcome = expected_outcomes[c_index]
        coalition_agendas = [np.multiply(agendas[c], poll_res[c]) for c in coalition]
        similarity = 0

        # Case 1: All issues weight equally
        if type == 1:
            for a in coalition_agendas:
                similarity += 2 - spatial.distance.cosine(expected_outcome, a)
        ratings[c_index] = similarity / (len(coalition)*2)

        # Case 2: Rate breaking point issues heavier
        # TODO

    r_sum = np.sum(ratings)
    ratings = ratings/r_sum

    return ratings


if __name__ == "__main__":
    poll_res = simulate_poll(3, 100)
    print("Poll result:\n{}".format(poll_res))
    # breaking_points = generate_breakingpoints(3, 10, 1)
    breaking_points = [[0], [0], [3]]
    print("Breaking points:\n{}".format(breaking_points))
    # agendas = generate_profile(5, 3)
    agendas = [[1, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]]
    print("Agendas:\n{}".format(agendas))
    possible_coaltitions = generate_coalitions(poll_res, agendas, breaking_points)
    print("Possible coalitions:\n{}".format(possible_coaltitions))
    expected_outcomes = simulate_outcomes(poll_res, possible_coaltitions, agendas, breaking_points)
    print("Expected Outcomes:\n{}".format(expected_outcomes))
    ratings = rate_coalitions(possible_coaltitions, agendas, expected_outcomes, breaking_points)
    print("Ratings:\n{}".format(ratings))
    final_expected_outcomes = np.sum([expected_outcomes[c] * ratings[c] for c in range(len(possible_coaltitions))], axis=0)
    print("Final Expected Outcomes:\n{}".format(final_expected_outcomes))
