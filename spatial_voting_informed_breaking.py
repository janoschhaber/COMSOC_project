# coding: utf-8

# In[1]:

### IMPORTANT: RESTART KERNEL AFTER UPDATING THE MODEL TO INVOKE CHANGES! ### 
from model import *
from plotting import *
from analysis import *
import pickle

print_model = True
print_analysis = True
revaluate = True

# selects breaking points

# In[8]:

def main():
    X = 4
    # number of voters
    N = 100
    # number of issues
    I_range = [5, 10, 20]
    B_range = [1,2,3,4,5,9,10,15,19,20]
    # number of breaking points
    random_breaking = False
    poll_results = [0.3, 0.3, 0.2, 0.2]

    I = 5
    B = 1

    print("X = {}, N = {}, I = {}, B = {}".format(X, N, I, B))

    agendas = generate_agendas(I, X)
    profile = generate_relative_profile(N, agendas, poll_results)

    vote_results, vote_supporters = simulate_vote(agendas, profile, 1)

    supporters = divide_electorate(profile, agendas)
    keys = list(supporters.keys())
    keys.sort()
    no_supporters = [len(supporters[key]) for key in keys]
    print("Poll results: {}, vote results: {}, supporters: {}".format(poll_results, vote_results, no_supporters))

    if random_breaking: breaking_points = generate_breakingpoints(X, I, B)
    else: breaking_points = derive_breaking_points(B, supporters, agendas)

    print("Breaking points: ", breaking_points)
    if revaluate == True:
        vote_results = revaluate_votes(agendas, profile, vote_supporters, vote_results, breaking_points)
        print("Revaluated votes: {}".format(vote_results))
    possible_coalitions = generate_coalitions(vote_results, agendas, breaking_points)
    if len(possible_coalitions) == 0:
        return

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


if __name__ == "__main__":
    main()


