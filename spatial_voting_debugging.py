### IMPORTANT: RESTART KERNEL AFTER UPDATING THE MODEL TO INVOKE CHANGES! ###
# get_ipython().magic('matplotlib inline')
from model import *
from plotting import *
from analysis import *
import pickle

print_model = True
print_analysis = True


# In[2]:

settings = [
#     X   I     N   C            W
    [ 4,  5, 10, "A", [0.3, 0.3, 0.2, 0.2]],
    [ 4, 10, 10, "A", [0.3, 0.3, 0.2, 0.2]],
    [ 4, 20, 10, "A", [0.3, 0.3, 0.2, 0.2]],
    [ 4,  5, 10, "B", [0.4, 0.4, 0.1, 0.1]],
    [ 4, 10, 10, "B", [0.4, 0.4, 0.1, 0.1]],
    [ 4, 20, 10, "B", [0.4, 0.4, 0.1, 0.1]],
    [ 6,  5, 10, "A", [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]],
    [ 6, 10, 10, "A", [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]],
    [ 6, 20, 10, "A", [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]],
    [ 6,  5, 10, "B", [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]],
    [ 6, 10, 10, "B", [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]],
    [ 6, 20, 10, "B", [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]],
    [10,  5, 10, "A", [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]],
    [10, 10, 10, "A", [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]],
    [10, 20, 10, "A", [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]],
    [10,  5, 10, "B", [0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]],
    [10, 10, 10, "B", [0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]],
    [10, 20, 10, "B", [0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]], 
]


# In[3]:

# E = 0: no breaking points
# E = 1: one random breaking point for the largest party
# E = 2: one breaking point on the top issue of the largest party
# E = 3: one random breaking point for the largest two parties
# E = 4: one breaking point on the top issues of the two largest party
# E = 5: one random breaking point per party
# E = 6: one top issue breaking point per party
# E = 7: two random breaking points per party
# E = 8: two top issue breaking points per party
# E = 9: one random breaking point for the smallest party
# E = 10: one breaking point on the top issue of the smallest party
# E = 11: one breaking point on the top issue of the first party + vote revaluation

for setting in settings[:3]:
    grid_results = []
    X, I, N, C, poll_results = setting
    print("X = {}, N = {}, I = {}, W = {}".format(X, N, I, poll_results))
    
    for i in range(1):             
        agendas = generate_agendas(I, X)

        if print_model: print("Agendas: ", agendas)

        supporters = generate_relative_profile(N, agendas, poll_results)
        supporter_IDs = get_supporter_IDs(supporters, X)
        no_supporters = [len(x_supp) for x_supp in supporters]
        profile = [item for sublist in supporters for item in sublist]
        
        # if print_model: print("Supporters: {}\nSupporter IDs:{}\nNumber of supporters:{}\nProfile: {}".format(supporters, supporter_IDs, no_supporters, profile) )
        
        vote_results = simulate_vote(agendas, profile)
        if print_model: print("Vote result: ", vote_results)

        median_voter = np.around(np.mean(profile, axis=0))
        if print_model: print("Median voter: ", median_voter)
        
        gold_agreement = np.mean(calculate_agreement(median_voter, profile, 1))
        if print_model: print("Gold agreement: ", gold_agreement)

        for E in range(12):
            # E = 0: no breaking points
            if E == 0: 
                breaking_points = generate_breakingpoints(X, I, 0)
            # E = 1: one random breaking point for the largest party
            # E = 3: one random breaking point for the largest two parties
            # E = 5: one random breaking point per party
            # E = 9: one random breaking point for the smallest party
            elif E == 1 or E == 3 or E == 5 or E == 9:
                breaking_points = generate_breakingpoints(X, I, 1)
            # E = 2: one breaking point on the top issue of the largest party
            # E = 4: one breaking point on the top issues of the two largest party
            # E = 6: one top issue breaking point per party
            # E = 10: one breaking point on the top issue of the smallest party
            # E = 11: one breaking point on the top issue of the first party + vote revaluation
            elif E == 2 or E == 4 or E == 6 or E == 10 or E == 11:
                breaking_points = derive_breaking_points(1, supporters, agendas)
            # E = 7: two random breaking points per party
            elif E == 7:
                breaking_points = generate_breakingpoints(X, I, 2)
            # E = 8: two top issue breaking points per party
            elif E == 8:
                breaking_points = derive_breaking_points(2, supporters, agendas)

            if E == 1 or E == 2 or E == 11:
                breaking_points_one = [None] * X
                breaking_points_one[0] = breaking_points[0]
                breaking_points = breaking_points_one

            if E == 3 or E == 4:
                breaking_points_one = [None] * X
                breaking_points_one[:1] = breaking_points[:1]
                breaking_points = breaking_points_one

            if E == 9 or E == 10:
                breaking_points_one = [None] * X
                breaking_points_one[-1] = breaking_points[-1]
                breaking_points = breaking_points_one
                
            if E == 11:
                vote_results = revaluate_votes(agendas, profile, supporter_IDs, vote_results, breaking_points)

            print("Experiment {}: Breaking points: {}".format(E, breaking_points))

            possible_coalitions = generate_coalitions(vote_results, agendas, breaking_points)
            if print_analysis: print("{} coalitions possible".format(len(possible_coalitions)))

            if len(possible_coalitions) == 0:
                grid_results.append([0, 0, 0, gold_agreement])
                if print_analysis: print("Expected agreement: 0 - No coalitions possible")
                continue

            expected_outcomes = simulate_outcomes(vote_results, possible_coalitions, agendas, breaking_points)
            # if print_model: print("Expected Outcomes: {}".format(expected_outcomes))
            ratings = rate_coalitions(possible_coalitions, agendas, vote_results, expected_outcomes, breaking_points)
            if math.isnan(ratings[0]):
                print("WARNING: Rating is NAN!")


            final_expected_outcomes = np.sum([expected_outcomes[c] * ratings[c] for c in range(len(possible_coalitions))], axis=0)
            if print_model: print("Expected Outcomes: {}".format(final_expected_outcomes))
            expected_agreements = calculate_agreement(final_expected_outcomes, profile, 1)
            if print_analysis: print("Expected agreement: ", np.mean(expected_agreements))

            if math.isnan(np.mean(expected_agreements)):
                print("WARNING: Expected agreement is NAN!")

            # # # SAMPLING # # #

            coalition_ID = form_coalition(ratings)
            # print(coalition_ID)
            coalition = possible_coalitions[coalition_ID]
            # print("Parties formed coalition {}".format(possible_coalitions[coalition_ID]))
            policy = form_policiy(coalition, agendas, vote_results, expected_outcomes[coalition_ID])
            # print("Coalition will implement policy {}".format(policy))
            gold_disagreements = np.sum(np.absolute(np.subtract(policy, median_voter)))
            gold_distance = I - 2 * gold_disagreements

            agreements = calculate_agreement(policy, profile, 1)
            if print_analysis: print("Sampled agreement: ", np.mean(agreements))

            entropy = calculate_entropy(final_expected_outcomes)
            if print_analysis: print("Entropy: ", entropy)

            grid_results.append([gold_agreement, np.mean(expected_agreements), np.mean(agreements), len(possible_coalitions), entropy])

            print(i, end='\r')

    filename = "model_runs/debugging/{}-{}-{}-{}.pickle".format(X, I, N, C)
    with open(filename, 'wb') as handle:
        pickle.dump(grid_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved.")
