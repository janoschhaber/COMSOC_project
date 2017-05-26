import numpy as np
import random as rd
from scipy import spatial
from itertools import chain, combinations
from copy import deepcopy


def powerset(iterable):
    """
    Generates the powerset of a list of items
    :param iterable: list of items 
    :return: powerset
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def simulate_poll(X, N, type=1, weightings=None):
    """
    Simulates a pre-election poll over X candidates with N voters. Type specifies the polling method
    :param X: number of candidates
    :param N: number of voters
    :param type: method used. 1: uniform, 2: random, 3: weighted distribution
    :param weightings: relative weight vector for all candidates X
    :return: vector of length X with normalised poll results
    """
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
    """
    Generates N profiles over I issues. Type specifies the method of profile generation
    :param I: number of issues
    :param N: number of agendas to return
    :param type: method used. 1: uniform, 2: equally weighted 3: individually weighted
    :param weight: specifies the weight for True on all (type 2) or each of the I issues (type 3)
    :return: matrix of NxI profiles
    """
    assert type < 3, "Profile generation error: Unknown generation type %i" % type
    profile = None

    # Case 1: Uniform distribution of preferences
    if type == 1:
        profile = np.array([np.random.choice([0, 1], size=(I,), p=[0.5, 0.5]) for _ in range(N)])

    # Case 2: Weighted distribution of preferences
    elif type == 2:
        assert len(weight) == 1, "Profile generation error: Issue weight inconsistent"
        profile = np.array([np.random.choice([0, 1], size=(I,), p=[weight, 1 - weight]) for _ in range(N)])

    elif type == 3:
        assert len(weight) == I, "Profile generation error: Issue weight vector inconsistent"
        # TODO
        pass

    return profile


def generate_breakingpoints(X, I, B, strict=True):
    """
    Generates B breaking points on I issues of the agenda of X candidates
    :param X: number of candidates
    :param I: nubmer of issues
    :param B: number of breaking points
    :param strict: True: exactly B breaking points. False: up to B breaking points
    :return: a list of X sublists with (up to) B breaking points on the I issues each
    """
    assert B <= I, "Breaking point error: Specified more breaking points than issues"
    breaking_points = [None] * X

    if B > 0:
        for c in range(X):
            b_issues = rd.sample(range(I), B)
            if not strict:
                remove = rd.randint(0, B)
                if remove > 0:
                    b_issues = b_issues[:remove]
            breaking_points[c] = b_issues

    return breaking_points


def generate_coalitions(poll_results, agendas, breaking_points):
    """
    Generates possible coalitions based on the poll results
    :param poll_results: results from the pull (relative weights of the candidates)
    :param agendas: agendas of the candidates
    :param breaking_points: breaking points specified by the candidates
    :return: a list of all possible coalitions
    """
    assert len(poll_results) == len(agendas)
    X = len(poll_results)
    coalitions = list()

    possible_coaltitions = powerset([c for c in range(X)])
    for coalition in possible_coaltitions:
        forced_issues = dict()
        consistent = True
        poll_representation = 0
        for c in coalition:
            if not consistent:
                break
            poll_representation += poll_results[c]
            if breaking_points[c] is not None:
                for b in breaking_points[c]:
                    if b in forced_issues and forced_issues[b] != agendas[c][b]:
                        consistent = False
                        break
                    elif b not in forced_issues:
                        forced_issues[b] = agendas[c][b]
        if consistent and poll_representation > 0.5:
            coalitions.append(coalition)

    return coalitions


def simulate_outcomes(poll_results, coalitions, agendas, breaking_points):
    """
    Weights the possible coalitions by the results of the poll to generate expected outcomes
    :param poll_results: results from the pull (relative weights of the candidates)
    :param coalitions: all possible coalitions
    :param agendas: agendas of the candidates
    :param breaking_points: breaking points specified by the candidates
    :return: a list of the expected outcomes of the I issues for each of the possible C coalitions
    """
    expected_outcomes = [None] * len(coalitions)

    for c_index, coalition in enumerate(coalitions):
        coalition_agendas = [np.multiply(agendas[c],poll_results[c]) for c in coalition]
        outcome = np.sum(coalition_agendas, axis=0)
        relative_weight = 0
        for c in coalition:
            relative_weight += poll_results[c]
        outcome = outcome / relative_weight

        for c in coalition:
            if breaking_points[c] is not None:
                c_breaking_points = breaking_points[c]
                for b in c_breaking_points:
                    outcome[b] = agendas[c][b]

        expected_outcomes[c_index] = outcome

    return expected_outcomes


def rate_coalitions(coalitions, agendas, poll_results, expected_outcomes, breaking_points, type = 1):
    """
    Rates all possible coalitions on their likelihood 
    based on the number of agreeing issues in the agendas of their members
    :param coalitions: all possible coalitions
    :param agendas: agendas of the candidates
    :param poll_results: results from the pull (relative weights of the candidates)
    :param expected_outcomes: expected outcomes of the I issues for each of the possible C coalitions
    :param breaking_points: breaking points specified by the candidates
    :param type: method of calculation. 1: all issues weight equal 2: breaking points are weighted heavier
    :return: a list of coalition ratings based on the number of agreeing issues in the agendas of their members 
    """
    I = len(expected_outcomes)
    ratings = np.zeros(len(coalitions))

    for c_index, coalition in enumerate(coalitions):
        expected_outcome = expected_outcomes[c_index]
        coalition_agendas = [np.multiply(agendas[c], poll_results[c]) for c in coalition]
        similarity = 0

        # Case 1: All issues weight equal
        if type == 1:
            for a in coalition_agendas:
                similarity += I - np.sum(np.absolute(np.subtract(expected_outcome, a)))
        # Case 2: Rate breaking point issues heavier
        else:
            # TODO
            pass

        ratings[c_index] = similarity / (len(coalition)*2)

    r_sum = np.sum(ratings)
    ratings = ratings/r_sum

    return ratings


def simulate_vote(agendas, electorate, type=1, special_interest=None):
    """
    Simulates a non-manipulated vote based on the parties' agendas and voters' preferences. 
    :param agendas: agendas of the candidates
    :param electorate: preferences of the voters
    :param type: method used. 1: Unweighted, 2: Voters have special interest topics weighted by special_interest
    :param special_interest: a vector of special interest weightings of the voters
    :return: 
    """
    X = len(agendas)
    I = len(agendas[0])
    N = len(electorate)
    votes = np.zeros(X)

    assert I == len(electorate[0]), "Number of issues of candidates and voters do not match!"

    for preference in electorate:
        # print("Preference is {}".format(preference))
        best_vote = None
        best_score = 0
        for p, agenda in enumerate(agendas):
            # Case 1: Voters have no special interest in issues
            if type == 1:
                score = I - np.sum(np.abs(np.subtract(preference, agenda)))
                if score > best_score:
                    best_vote = p
                    best_score = score
            # Case 2: Voters have special interests in some issues
            else:
                # TODO
                pass

        if best_vote is None:
            best_vote = rd.randint(0, X)

        votes[best_vote] += 1
        # print("Best match is {} with an agreement of {}".format(agendas[best_vote], best_score))

    return votes / N


def form_coalition(ratings):
    """
    Returns a coalition based on the ratings obtained from a vote
    :param possible_coaltitions: all possible coalitions
    :param ratings: a list of coalition ratings based on the number of agreeing issues in the agendas of their members
    :return: a coalition ID based on the ratings obtained from a vote
    """
    spectrum = np.cumsum(ratings)
    index = np.random.random_sample()
    for i, c in enumerate(spectrum):
        if index <= c:
            return i

    return -1


def form_policiy(coalition, agendas, vote_results, outcome_probs):
    """
    
    :param coalition: 
    :param agendas: 
    :param vote_results: 
    :param outcome_probs: 
    :return: 
    """
    open_issues = []
    policy = deepcopy(outcome_probs)
    print(policy)

    for i, prob in enumerate(outcome_probs):
        if prob != 0 and prob != 1:
            open_issues.append(i)

    # print("{} open issues". format(len(open_issues)))
    if len(open_issues) == 0:
        return policy
    else:
        candidate_issues = {}
        for i, candidate in enumerate(coalition):
            candidate_issues[i] = int(vote_results[candidate] * len(open_issues))

        # print("{} issues are divided based on the vote weights".format(np.sum(list(candidate_issues.values()))))
        for candidate, no_issues in candidate_issues.items():
            for _ in range(no_issues):
                open_issue = open_issues.pop()
                policy[open_issue] = agendas[candidate][open_issue]


        if len(open_issues) == 0:
            return policy
        else:
            # print("{} remaining issues decided by weighted coin toss".format(len(open_issues)))
            spectrum = np.cumsum([vote_results[candidate] for candidate in coalition])
            spectrum = spectrum/np.max(spectrum)
            print(spectrum)
            for open_issue in open_issues:
                index = np.random.random_sample()
                for i, c in enumerate(spectrum):
                    if index <= c:
                        policy[open_issue] = agendas[coalition[i]][open_issue]


            return policy
