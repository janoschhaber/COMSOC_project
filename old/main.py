from model import *
from plotting import *
from analysis import *
import pickle

print_model = True
print_analysis = True

def main():


    X = 4
    # number of voters
    N = 100
    # number of issues
    I = 5
    # number of breaking points
    B = 1
    print("X = {}, N = {}, I = {}, B = {}".format(X, N, I, B))

    poll_results = [0.3, 0.3, 0.2, 0.2]

    grid_results = []

    for i in range(1000):
        agendas = generate_profile(I, X)
        profile = generate_relative_profile(N, agendas, poll_results)

        vote_results, _ = simulate_vote(agendas, profile, 1)
        # print("Poll results: {}, vote results: {}".format(poll_results, vote_results))

        breaking_points = generate_breakingpoints(X, I, B)
        possible_coalitions = generate_coalitions(vote_results, agendas, breaking_points)

        # print(possible_coalitions)
        if len(possible_coalitions) == 0:
            grid_results.append([-1, I, -I, 1])
            continue

        expected_outcomes = simulate_outcomes(vote_results, possible_coalitions, agendas, breaking_points)
        ratings = rate_coalitions(possible_coalitions, agendas, vote_results, expected_outcomes, breaking_points)
        final_expected_outcomes = np.sum([expected_outcomes[c] * ratings[c] for c in range(len(possible_coalitions))],
                                         axis=0)
        # if print_model: print("Final Expected Outcomes:\n{}".format(final_expected_outcomes))
        coalition_ID = form_coalition(ratings)
        # print(coalition_ID)

        coalition = possible_coalitions[coalition_ID]
        # print("Parties formed coalition {}".format(possible_coaltitions[coalition_ID]))
        policy = form_policiy(coalition, agendas, vote_results, expected_outcomes[coalition_ID])
        # print("Coalition will implement policy {}".format(policy))

        # Sanity check:
        for index, entry in enumerate(expected_outcomes[coalition_ID]):
            if entry == 1.0:
                assert policy[index] == 1.0, "SANITY CHECK FAILED! Breaking point violated"
            if entry == 0.0:
                assert policy[index] == 0.0, "SANITY CHECK FAILED! Breaking point violated"

        regrets = calculate_regret(policy, profile, 2)
        # print("Total regret is {}, with mean {}, max of {} and min of {}. Std is {}".format(np.sum(regrets), np.mean(regrets), np.max(regrets), np.min(regrets), np.std(regrets)))
        grid_results.append([np.mean(regrets), np.max(regrets), np.min(regrets), np.std(regrets)])

      #  print(i, end='\r')

    # profile = generate_profile(I, N)
    # # print("profile generated {} preference lists".format(len(profile)))
    #
    # poll_results = simulate_poll(X, N)
    # print("Poll result:\n{}".format(poll_results))
    #
    # agendas = generate_profile(I, X)
    # # agendas = np.array([[1, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]])
    # print("Agendas:\n{}".format(agendas))
    #
    # # without breaking points
    #
    # no_breaking_points = generate_breakingpoints(X, I, 0)
    # if print_model: print("\nNo breaking points")
    #
    # possible_coaltitions = generate_coalitions(poll_results, agendas, no_breaking_points)
    # # if print_model: print("Possible coalitions:\n{}".format(possible_coaltitions))
    # expected_outcomes = simulate_outcomes(poll_results, possible_coaltitions, agendas, no_breaking_points)
    # # if print_model: print("Expected Outcomes:\n{}".format(expected_outcomes))
    # ratings = rate_coalitions(possible_coaltitions, agendas, poll_results, expected_outcomes, no_breaking_points)
    # # if print_model: print("Ratings:\n{}".format(ratings))
    # final_expected_outcomes = np.sum([expected_outcomes[c] * ratings[c] for c in range(len(possible_coaltitions))], axis=0)
    # likely_coalition = possible_coaltitions[form_coalition(ratings)]
    # if print_model: print("Final Expected Outcomes:\n{}".format(final_expected_outcomes))
    #
    # # with breaking points
    #
    # breaking_points = generate_breakingpoints(X, I, 1)
    # # breaking_points = [[0], [0], [3]]
    # if print_model: print("\nBreaking points: {}".format(breaking_points))
    #
    # b_possible_coaltitions = generate_coalitions(poll_results, agendas, breaking_points)
    # # if print_model: print("Possible coalitions:\n{}".format(b_possible_coaltitions))
    # b_expected_outcomes = simulate_outcomes(poll_results, b_possible_coaltitions, agendas, breaking_points)
    # # if print_model: print("Expected Outcomes:\n{}".format(b_expected_outcomes))
    # b_ratings = rate_coalitions(b_possible_coaltitions, agendas, poll_results, b_expected_outcomes, breaking_points)
    # # if print_model: print("Ratings:\n{}".format(b_ratings))
    # b_final_expected_outcomes = np.sum([b_expected_outcomes[c] * b_ratings[c] for c in range(len(b_possible_coaltitions))], axis=0)
    # b_likely_coalition = b_possible_coaltitions[form_coalition(b_ratings)]
    # if print_model: print("Final Expected Outcomes:\n{}".format(b_final_expected_outcomes))
    #
    # # analysis
    #
    # c_difference = len(possible_coaltitions) - len(b_possible_coaltitions)
    # if print_analysis: print("\n{} coalitions became inconsistent by introducing breaking points".format(c_difference))
    #
    # l_differences, l_total_diff, l_avg_diff = calculate_likelihood_diff(final_expected_outcomes, b_final_expected_outcomes)
    # print("Likelihoods changed by {}, with a total of {}, averaging {} per issue.".format(l_differences, l_total_diff, l_avg_diff))
    #
    # # plot_bar_chart(pd.DataFrame([final_expected_outcomes, b_final_expected_outcomes]).T)
    #
    # entropy = calculate_entropy(final_expected_outcomes)
    # b_entroy = calculate_entropy(b_final_expected_outcomes)
    # if entropy > b_entroy:
    #     print("Entropy decreased by {} from {} to {}".format(entropy-b_entroy, entropy, b_entroy))
    # elif entropy < b_entroy:
    #     print("Entropy increased by {} from {} to {}".format(b_entroy - entropy, entropy, b_entroy))
    # else:
    #     print("Entropy remained unchanged at {}".format(entropy))
    #
    # # vote
    #
    # vote_results = simulate_vote(agendas, profile, 3, None, None, possible_coaltitions, likely_coalition)
    # b_vote_results =  simulate_vote(agendas, profile, 3, None, None, b_possible_coaltitions, b_likely_coalition)
    # print("\nVote result: {}".format(vote_results))
    #
    # # without breaking points
    #
    # if print_model: print("\nNo breaking points")
    #
    # possible_coaltitions = generate_coalitions(vote_results, agendas, no_breaking_points)
    # # if print_model: print("Possible coalitions:\n{}".format(possible_coaltitions))
    # expected_outcomes = simulate_outcomes(vote_results, possible_coaltitions, agendas, no_breaking_points)
    # # if print_model: print("Expected Outcomes:\n{}".format(expected_outcomes))
    # ratings = rate_coalitions(possible_coaltitions, agendas, vote_results, expected_outcomes, no_breaking_points)
    # # if print_model: print("Ratings:\n{}".format(ratings))
    # coalition_ID = form_coalition(ratings)
    # coalition = possible_coaltitions[coalition_ID]
    # print("Parties formed coalition {}".format(possible_coaltitions[coalition_ID]))
    #
    # policy = form_policiy(coalition, agendas, vote_results, expected_outcomes[coalition_ID])
    # print("Coalition will implement policy {}".format(policy))
    #
    # # Sanity check:
    # for index, entry in enumerate(expected_outcomes[coalition_ID]):
    #     if entry == 1.0:
    #         assert policy[index] == 1.0, "SANITY CHECK FAILED! Breaking point violated"
    #     if entry == 0.0:
    #         assert policy[index] == 0.0, "SANITY CHECK FAILED! Breaking point violated"
    #
    # regrets = calculate_regret(policy, profile)
    # print("Total regret is {}, with mean {}, max of {} and min of {}. Std is {}".format(np.sum(regrets), np.mean(regrets), np.max(regrets), np.min(regrets), np.std(regrets)))
    #
    # # with breaking points
    #
    # b_possible_coaltitions = generate_coalitions(vote_results, agendas, breaking_points)
    # # if print_model: print("Possible coalitions:\n{}".format(b_possible_coaltitions))
    # b_expected_outcomes = simulate_outcomes(vote_results, b_possible_coaltitions, agendas, breaking_points)
    # # if print_model: print("Expected Outcomes:\n{}".format(b_expected_outcomes))
    # b_ratings = rate_coalitions(b_possible_coaltitions, agendas, vote_results, b_expected_outcomes, breaking_points)
    # # if print_model: print("Ratings:\n{}".format(b_ratings))
    #
    # b_coalition_ID = form_coalition(b_ratings)
    # b_coalition = possible_coaltitions[b_coalition_ID]
    # print("Parties formed coalition {}".format(possible_coaltitions[b_coalition_ID]))
    #
    # b_policy = form_policiy(b_coalition, agendas, vote_results, b_expected_outcomes[b_coalition_ID])
    # print("Coalition will implement policy {}".format(b_policy))
    #
    # # Sanity check:
    # for index, entry in enumerate(b_expected_outcomes[b_coalition_ID]):
    #     if entry == 1.0:
    #         assert b_policy[index] == 1.0, "SANITY CHECK FAILED! Breaking point violated"
    #     if entry == 0.0:
    #         assert b_policy[index] == 0.0, "SANITY CHECK FAILED! Breaking point violated"
    #
    # b_regrets = calculate_regret(b_policy, profile)
    # print(
    #     "Total regret is {}, with mean {}, max of {} and min of {}. Std is {}".format(np.sum(b_regrets), np.mean(b_regrets),
    #                                                                                   np.max(b_regrets), np.min(b_regrets),
    #                                                                                   np.std(b_regrets)))

if __name__ == "__main__":
    main()