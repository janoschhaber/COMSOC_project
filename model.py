import numpy as np
import random as rd
from itertools import chain, combinations
from copy import deepcopy
from collections import defaultdict
import math


def powerset(iterable):
    """
    Generates the powerset of a list of items
    :param iterable: list of items 
    :return: powerset
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_supporter_IDs(supporters, X):
    supporter_IDs = [None] * X
    first_ID = 0
    for i, x_supp in enumerate(supporters):
        IDs = [first_ID + i for i in range(len(x_supp))]
        supporter_IDs[i] = IDs
        first_ID += len(x_supp)
    return supporter_IDs


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
    :return: matrix of NxI preferences
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


def generate_agendas(I, X):
    agendas = list()

    for _ in range(X):
        agenda = list(np.random.choice([0, 1], size=(I,), p=[0.5, 0.5]))
        while agenda in agendas:
            agenda = list(np.random.choice([0, 1], size=(I,), p=[0.5, 0.5]))
        agendas.append(agenda)

    return agendas


def generate_relative_profile(N, agendas, weights):
    """
    Generates an electorate profile according to specified weights of the candidates in an election
    :param N: number of preferences (size of the profile)
    :param agendas: the agendas of the candidates
    :param weights: the relative weights of each of the candidates in the election
    :return: list of Nx preferences over I issues for all candidates x in X
    """
    assert len(agendas) == len(weights), "Profile generation error: Candidate weight vector inconsistent"
    assert sum(weights) > 0.95, "Profile generation error: Weights are underspecified"
    assert sum(weights) < 1.05, "Profile generation error: Weights sum up to more than 1"

    votes = [int(N*w) for w in weights]
    profile = [[] for i in range(len(agendas))]
    I = len(agendas[0])
    no_generated = 0

    for i, a in enumerate(agendas):
        for _ in range(votes[i]):
            preference = np.random.choice([0, 1], size=(I,), p=[0.5, 0.5])
            while not is_closest(preference, i, agendas):
                preference = swap_one(preference, a)
            profile[i].append(preference)
            no_generated += 1

    if no_generated != N:
        # print("{} remaining preferences generated randomly.".format(N-no_generated))
        for _ in range(N-no_generated):
            profile[i].append(np.random.choice([0, 1], size=(I,), p=[0.5, 0.5]))

    return profile


def is_closest(preference, candidate, agendas):
    """
    Returns True if a given preference is closest to the agenda of a specified candidate 
    :param preference: a certain voter preference
    :param candidate: the candidate to compare to
    :param agendas: the full list of candidate agendas
    :return: True if a given preference is closest to the agenda of a specified candidate, False otherwise
    """
    I = len(agendas[0])
    best_vote = None
    best_score = -I
    # print("Preference is ", preference)
    # print("Agenda is ", agendas[candidate])

    if np.array_equal(preference, agendas[candidate]): return True

    for p, agenda in enumerate(agendas):
        # Case 1: Voters have no special interest in issues
        disagreements = np.sum(np.absolute(np.subtract(preference, agenda)))
        score = I - 2 * disagreements
        if score > best_score:
            best_vote = p
            best_score = score


    if best_vote == candidate:
        # print("Best vote is candidate!")
        return True
    else:
        # print("Best vote is agenda ", agendas[best_vote])
        return False


def swap_one(preference, agenda):
    """
    Swaps one random issue in a preference vector to move the preference closer towards a specified agenda
    :param preference: the initial voter preference 
    :param agenda: the goal agenda
    :return: the preference vector with one element swapped
    """
    indexes = np.random.permutation(len(preference))
    for i in indexes:
        p = preference[i]
        if p != agenda[i]:
            preference[i] = agenda[i]
            return preference

    print("WARNING: Vote could not be converted to match agenda!")
    return preference


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
    :param poll_results: results from the poll (relative weights of the candidates)
    :param coalitions: all possible coalitions
    :param agendas: agendas of the candidates
    :param breaking_points: breaking points specified by the candidates
    :return: a list of the expected outcomes of the I issues for each of the possible C coalitions
    """
    expected_outcomes = [None] * len(coalitions)

    for c_index, coalition in enumerate(coalitions):
        coalition_agendas = [np.multiply(agendas[c], poll_results[c]) for c in coalition]
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
    I = len(expected_outcomes[0])
    ratings = np.zeros(len(coalitions))

    for c_index, coalition in enumerate(coalitions):
        expected_outcome = expected_outcomes[c_index]
        coalition_agendas = [np.multiply(agendas[c], poll_results[c]) for c in coalition]
        similarity = 0

        for a in coalition_agendas:
            # Case 1: Simple distance
            if type == 1:
                # print("Agenda: {}, expected_outcome: {}, similiarity: {}".format(a, expected_outcome, (I - np.sum(np.absolute(np.subtract(expected_outcome, a)))) / I))
                similarity += (I - np.sum(np.absolute(np.subtract(expected_outcome, a)))) / I
            # Case 2: Absolute difference
            elif type == 2:
                disagreements = np.sum(np.absolute(np.subtract(expected_outcome, a)))
                similarity += I - 2 * disagreements
            # Case 3: Breaking points are weighted heavier
            else:
                # TODO
                pass

        ratings[c_index] = similarity / (len(coalition))

    r_sum = np.sum(ratings)
    if r_sum == 0:
        print(coalitions, agendas, breaking_points, expected_outcomes[0], coalition_agendas)
        r_sum = 1
    ratings = ratings/r_sum

    return ratings

def revaluate_votes(agendas, profile, supporters, vote_results, breaking_points):
    """
       Recalculates the votes assuming that if a breaking point doesnt coincide with
       the suporters value it will change to the one that does and preferably the one closest
       to him. If there is none that does then the voter's vote remains in the current party.
       :param agendas: agendas of the candidates
       :param profle: preferences of the voters
       :param vote_results: current votes
       :param breaking_points: list of breaking points implemented
       :return:
       """
    for party, b_point in enumerate(breaking_points):
        if b_point!=None:
            change_party=[]
            #TODO model when there are more than one breaking point per party
            issue = b_point[0]
            for supporter in supporters[party]:
                if profile[supporter][issue]!=agendas[party][issue]:
                    # try to change vote
                    change_party.append(supporter)
            if len(change_party) > 0:
                vote_results = change_votes(change_party, party, profile, agendas, issue, vote_results, len(profile))
    return vote_results



def change_votes(change_party, party, profile, agendas, issue, vote_results,n):
    #Get the possible changes
    I = len(agendas[0])
    vote_results = n * vote_results
    #Assuming that there is only one breaking point per party, thus all supporters need the same thing, else move this inside loop
    remaining_agendas = [p for p, x in enumerate(agendas) if x[issue] == profile[change_party[0]][issue]]
    for supporter in change_party:
        best_vote = None
        best_score = -I
        for p in remaining_agendas:
            disagreements = np.sum(np.absolute(np.subtract(profile[supporter], agendas[p])))
            score = I - 2 * disagreements
            if score > best_score:
                best_vote = p
                best_score = score
        if best_vote != None:
            # recalculate vote results
            vote_results[party]-=1
            vote_results[p] +=1
    return vote_results/n


def simulate_vote(agendas, electorate):
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
    wasted_votes = 0

    # print("Likely coalition: ", likely_coalition)

    assert I == len(electorate[0]), "Number of issues of candidates and voters do not match!"

    for voter, preference in enumerate(electorate):
        # print("Preference is {}".format(preference))
        best_vote = None
        best_score = -I
        for p, agenda in enumerate(agendas):
            disagreements = np.sum(np.absolute(np.subtract(preference, agenda)))
            score = I - 2 * disagreements
            if score > best_score:
                best_vote = p
                best_score = score

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
    Determines a policy based on the coalition formed and the agendas of its members
    :param coalition: the coalition that was formed 
    :param agendas: the agendas of the coalition partners
    :param vote_results: the results of the vote
    :param outcome_probs: the outcome probabilities as expected based on the vote
    :return: the policy to be implemented by the coalition
    """
    open_issues = []
    policy = deepcopy(outcome_probs)
    # print(policy)

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
            # print(spectrum)
            for open_issue in open_issues:
                index = np.random.random_sample()
                for i, c in enumerate(spectrum):
                    if index <= c:
                        policy[open_issue] = agendas[coalition[i]][open_issue]


            return policy


def divide_electorate(electorate, agendas):
    """
    Divides the electorate into supporter groups of the different candidates
    :param electorate: the electorate
    :param agendas: the agendas of the candidates
    :return: a list of supporting preferences for each of the candidates
    """
    X = len(agendas)
    I = len(agendas[0])
    support = defaultdict(list)

    assert I == len(electorate[0]), "Number of issues of candidates and voters do not match!"

    for preference in electorate:
        best_vote = None
        best_score = -I
        for p, agenda in enumerate(agendas):
            disagreements = np.sum(np.absolute(np.subtract(preference, agenda)))
            score = I - 2 * disagreements
            if score > best_score:
                best_vote = p
                best_score = score

        if best_vote is None:
            best_vote = rd.randint(0, X-1)

        support[best_vote].append(preference)

    return support


def derive_breaking_points(B, supporters, agendas):
    """
    Derives a candidate's breaking point(s) from the mean voter's profile
    :param B: the number of breaking points to be stated
    :param supporters: the preferences of the voters that voted for a given candidate
    :param agendas: the agendas of the different candidates
    :return: a list of breaking point issue indexes per candidate
    """
    I = len(supporters[0][0])
    assert B <= I, "Breaking point error: Specified more breaking points than issues"
    X = len(supporters)
    breaking_points = [None] * X

    if B == 0:
         return breaking_points

    one_vector = np.ones(I)
    for c, c_sup in enumerate(supporters):

        if len(c_sup) == 0:
            breaking_points[c] = rd.sample(range(I), B)
        else:
            mean_vote = np.mean(c_sup, axis=0)
            median_vote = np.around(mean_vote)

            # print("Party agenda: {}, Mean vote: {}".format(agendas[c], mean_vote))
            distance_from_one = np.subtract(one_vector, mean_vote)
            agreement_vector = np.minimum(mean_vote, distance_from_one)
            # print("Agreement vector: ", agreement_vector)
            sorted_agreement = np.argsort(agreement_vector)
            # print("Sorted agreement: ", sorted_agreement)
            # TODO: Introduce some exponential decaying randomness
            c_breaking_points = []
            agreement_intersection = [i for i in range(I) if agendas[c][i] == median_vote[i] ]

            if len(agreement_intersection) >= B:
                for priority_issue in sorted_agreement:
                    if priority_issue in agreement_intersection:
                        c_breaking_points.append(priority_issue)
                    if len(c_breaking_points) == B: break
            else:
                print("WARNING: Setting more breaking points than agreement issues")

            # print("Sorted agreement is {}, breaking points are set on {}".format(sorted_agreement, c_breaking_points))
            breaking_points[c] = c_breaking_points
    return breaking_points
