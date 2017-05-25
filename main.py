from model import *
from plotting import *
from analysis import *

print_model = True
print_analysis = True

def main():
    X = 3
    # number of voters
    N = 100
    # number of issues
    I = 5
    # number of breaking points
    B = 1
    print("X = {}, N = {}, I = {}, B = {}".format(X, N, I, B))

    poll_results = simulate_poll(X, N)
    print("Poll result:\n{}".format(poll_results))
    # agendas = generate_profile(5, 3)
    agendas = np.array([[1, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]])
    print("Agendas:\n{}".format(agendas))

    # without breaking points

    breaking_points = generate_breakingpoints(X, I, 0)

    possible_coaltitions = generate_coalitions(poll_results, agendas, breaking_points)
    if print_model: print("Possible coalitions:\n{}".format(possible_coaltitions))
    expected_outcomes = simulate_outcomes(poll_results, possible_coaltitions, agendas, breaking_points)
    if print_model: print("Expected Outcomes:\n{}".format(expected_outcomes))
    ratings = rate_coalitions(possible_coaltitions, agendas, poll_results, expected_outcomes, breaking_points)
    if print_model: print("Ratings:\n{}".format(ratings))
    final_expected_outcomes = np.sum([expected_outcomes[c] * ratings[c] for c in range(len(possible_coaltitions))], axis=0)
    if print_model: print("Final Expected Outcomes:\n{}".format(final_expected_outcomes))

    # with breaking points

    # breaking_points = generate_breakingpoints(3, 10, 1)
    breaking_points = [[0], [0], [3]]
    if print_model: print("\nBreaking points:\n{}".format(breaking_points))

    b_possible_coaltitions = generate_coalitions(poll_results, agendas, breaking_points)
    if print_model: print("Possible coalitions:\n{}".format(b_possible_coaltitions))
    b_expected_outcomes = simulate_outcomes(poll_results, b_possible_coaltitions, agendas, breaking_points)
    if print_model: print("Expected Outcomes:\n{}".format(b_expected_outcomes))
    b_ratings = rate_coalitions(b_possible_coaltitions, agendas, poll_results, b_expected_outcomes, breaking_points)
    if print_model: print("Ratings:\n{}".format(b_ratings))
    b_final_expected_outcomes = np.sum([b_expected_outcomes[c] * b_ratings[c] for c in range(len(b_possible_coaltitions))], axis=0)
    if print_model: print("Final Expected Outcomes:\n{}".format(b_final_expected_outcomes))


    # analysis
    c_difference = len(possible_coaltitions) - len(b_possible_coaltitions)
    if print_analysis: print("\n{} coalitions became inconsistent by introducing breaking points".format(c_difference))

    l_differences, l_total_diff, l_avg_diff = calculate_likelihood_diff(final_expected_outcomes, b_final_expected_outcomes)

    print("Likelihoods changed by {}, with a total of {}, averaging {} per issue.".format(l_differences, l_total_diff, l_avg_diff))

    # plot_bar_chart(pd.DataFrame([final_expected_outcomes, b_final_expected_outcomes]).T)

    entropy = calculate_entropy(final_expected_outcomes)
    b_entroy = calculate_entropy(b_final_expected_outcomes)

    #entropy = calculate_entropy([0.5, 0.5, 0.5, 0.5, 0.5])
    #b_entroy = calculate_entropy([1, 1, 1, 1, 1])

    if entropy > b_entroy:
        print("Entropy decreased by {} from {} to {}".format(entropy-b_entroy, entropy, b_entroy))
    elif entropy < b_entroy:
        print("Entropy increased by {} from {} to {}".format(b_entroy - entropy, entropy, b_entroy))
    else:
        print("Entropy remained unchanged at {}".format(entropy))

    electorate = generate_profile(I, N)
    # print(electorate)

    print("Electorate generated {} preference lists".format(len(electorate)))

    vote_results = simulate_vote(agendas, electorate)

    print("Vote result: {}".format(vote_results))

    # without breaking points

    breaking_points = generate_breakingpoints(X, I, 0)

    possible_coaltitions = generate_coalitions(vote_results, agendas, breaking_points)
    if print_model: print("Possible coalitions:\n{}".format(possible_coaltitions))
    expected_outcomes = simulate_outcomes(vote_results, possible_coaltitions, agendas, breaking_points)
    if print_model: print("Expected Outcomes:\n{}".format(expected_outcomes))
    ratings = rate_coalitions(possible_coaltitions, agendas, vote_results, expected_outcomes, breaking_points)
    if print_model: print("Ratings:\n{}".format(ratings))
    coalition_ID = form_coalition(ratings)
    coalition = possible_coaltitions[coalition_ID]
    print("Parties formed coalition {}".format(possible_coaltitions[coalition_ID]))

    policy = form_policiy(coalition, agendas, vote_results, expected_outcomes[coalition_ID])
    print("Coalition will implement policy {}".format(policy))
    print("Expected outcome was: {}".format(expected_outcomes[coalition_ID]))

    # with breaking points

    # breaking_points = generate_breakingpoints(3, 10, 1)
    breaking_points = [[0], [0], [3]]
    if print_model: print("\nBreaking points:\n{}".format(breaking_points))

    b_possible_coaltitions = generate_coalitions(vote_results, agendas, breaking_points)
    if print_model: print("Possible coalitions:\n{}".format(b_possible_coaltitions))
    b_expected_outcomes = simulate_outcomes(vote_results, b_possible_coaltitions, agendas, breaking_points)
    if print_model: print("Expected Outcomes:\n{}".format(b_expected_outcomes))
    b_ratings = rate_coalitions(b_possible_coaltitions, agendas, vote_results, b_expected_outcomes, breaking_points)
    if print_model: print("Ratings:\n{}".format(b_ratings))

    b_coalition_ID = form_coalition(b_ratings)
    b_coalition = possible_coaltitions[b_coalition_ID]
    print("Parties formed coalition {}".format(possible_coaltitions[b_coalition_ID]))

    b_policy = form_policiy(b_coalition, agendas, vote_results, b_expected_outcomes[b_coalition_ID])
    print("Coalition will implement policy {}".format(b_policy))
    print("Expected outcome was: {}".format(b_expected_outcomes[b_coalition_ID]))

if __name__ == "__main__":
    main()