import rg

dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
               [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]

minsup = 0.15
minconf = 0.6
cars = rg.rule_generator(dataset, minsup, minconf)

# print("CARs:")
# cars.print_rule()

# print("prCARs:")
# cars.prune_rules(dataset)
# cars.print_pruned_rule()