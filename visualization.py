import pickle
import numpy as np

# grid_results: [gold_agreement, expected_agreement, sampled_agreement, no_possible_coaltions, entropy]
E = 12

grid_results = None
files = ["4-5-10-A.pickle", "4-10-10-A.pickle",
         "4-20-10-A.pickle"]

file = files[0]
with open('model_runs/debugging/' + file, 'rb') as handle:
    grid_results = pickle.load(handle)

avg_gold_regrets = [None] * E
avg_expected_regrets = [None] * E
avg_sampled_regrets = [None] * E
avg_possible_coalitions = [None] * E
avg_entropies = [None] * E
normalized_expected_regrets = [None] * E
normalized_sampled_regrets = [None] * E
expected_regrets_std = [None] * E
sampled_regrets_std = [None] * E

labels = [n for n in range(E)]

no_results = len(grid_results)
print(no_results)

for e in range(E):
    gold_regrets = [None] * int(no_results / E)
    expected_regrets = [None] * int(no_results / E)
    sampled_regrets = [None] * int(no_results / E)
    no_possible_coalitions = [None] * int(no_results / E)
    entropies = [None] * int(no_results / E)

    for i, index in enumerate(range(e, len(grid_results), E)):
        gold_regrets[i], expected_regrets[i], sampled_regrets[i], no_possible_coalitions[i], entropies[i] = \
        grid_results[index]

    avg_gold_regrets[e] = np.sum(gold_regrets) / len(gold_regrets)

    # TODO: Check if we need this line
    # expected_regrets = [value if value > 0 else 0 for value in expected_regrets]
    avg_expected_regrets[e] = np.sum(expected_regrets) / len(expected_regrets)

    avg_sampled_regrets[e] = np.sum(sampled_regrets) / len(sampled_regrets)

    avg_possible_coalitions[e] = np.sum(no_possible_coalitions) / len(no_possible_coalitions)

    avg_entropies[e] = np.sum(entropies) / len(entropies)

    if avg_gold_regrets[e] == 0:
        gold_regret = 0.1
    else:
        gold_regret = avg_gold_regrets[e]

    normalized_expected_regrets[e] = avg_expected_regrets[e] / gold_regret
    normalized_sampled_regrets[e] = avg_sampled_regrets[e] / gold_regret

    expected_regrets_std[e] = np.std(expected_regrets)
    sampled_regrets_std[e] = np.std(sampled_regrets)


print(avg_gold_regrets)
print(avg_expected_regrets)
print(avg_sampled_regrets)
print(avg_possible_coalitions)
print(avg_entropies)