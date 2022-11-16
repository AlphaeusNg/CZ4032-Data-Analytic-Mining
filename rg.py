from ruleitem import RuleItem
from car import CAR
from frequentruleitem import FrequentRuleItems

def join(item1, item2, dataset):
    cat1 = set(item1.cond_set)
    cat2 = set(item2.cond_set)

    if cat1 == cat2 or item1.class_label != item2.class_label:
        return None

    intersect = cat1.intersection(cat2)
    
    for item in intersect:
        if item1.cond_set[item] != item2.cond_set[item]:
            return None

    category = cat1 or cat2

    new_cond_set = dict()

    for item in category:
        if item in cat1:
            new_cond_set[item] = item1.cond_set[item]
        else:
            new_cond_set[item] = item2.cond_set[item]

    return RuleItem(new_cond_set, item1.class_label, dataset)


def candidate_gen(frequent_ruleitems, dataset):
    ret_frequentruleitems = FrequentRuleItems()

    for item1 in frequent_ruleitems.frequent_ruleitems_set:
        for item2 in frequent_ruleitems.frequent_ruleitems_set:
            new_ruleitem = join(item1, item2, dataset)
            if new_ruleitem != None:
                ret_frequentruleitems.add(new_ruleitem)
                if ret_frequentruleitems.getSize() >= 80000:      # not allow to store more than 1000 ruleitems
                    return ret_frequentruleitems

    return ret_frequentruleitems


def rule_generator(dataset, minsup, minconf):
    frequent_ruleitems = FrequentRuleItems()
    _car = CAR()

    class_label = set([x[-1] for x in dataset])
    for column in range(0, len(dataset[0])-1):
        distinct_value = set([x[column] for x in dataset])
        for value in distinct_value:
            cond_set = {column: value}
            for classes in class_label:
                rule_item = RuleItem(cond_set, classes, dataset)
                if rule_item.sup >= minsup:
                    frequent_ruleitems.add(rule_item)
    _car.generate_rules(frequent_ruleitems, minsup, minconf)
    cars = _car

    last_cars_number = 0
    curr_cars_num = len(cars.rules)
    while frequent_ruleitems.getSize() > 0 and curr_cars_num <= 2000 and (curr_cars_num - last_cars_number) >= 10:
        candidate = candidate_gen(frequent_ruleitems, dataset)
        frequent_ruleitems = FrequentRuleItems()
        _car = CAR()
        for item in candidate.frequent_ruleitems_set:
            if item.sup >= minsup:
                frequent_ruleitems.add(item)
        _car.generate_rules(frequent_ruleitems, minsup, minconf)
        cars.append(_car, minsup, minconf)
        last_cars_number = curr_cars_num
        curr_cars_num = len(cars.rules)

    return cars