from ruleitem import RuleItem
import sys


class CAR:
    def __init__(self):
        self.rules = set()
        self.pruned_rules = set()

    def print_rule(self):
        for item in self.rules:
            item.print_rule()

    def print_pruned_rule(self):
        for item in self.pruned_rules:
            item.print_rule()

    def _add(self, rule_item, minSup, minConf):
        if rule_item.sup >= minSup and rule_item.conf >= minConf:
            if rule_item in self.rules:
                return
            # Prune rules of same condition set, but lower confidence
            for item in self.rules:
                if item.cond_set == rule_item.cond_set:  
                    if item.conf < rule_item.conf:
                        self.rules.remove(item)
                        self.rules.add(rule_item)
                    return
            self.rules.add(rule_item)

    def generate_rules(self, freq_ruleitems, minSup, minConf):
        for item in freq_ruleitems.freq_ruleitems_set:
            self._add(item, minSup, minConf)

    def prune_rules(self, dataset):
        for rule in self.rules:
            pruned_rule = prune(rule, dataset)

            is_existed = False
            for rule in self.pruned_rules:
                if rule.class_label == pruned_rule.class_label:
                    if rule.cond_set == pruned_rule.cond_set:
                        is_existed = True
                        break

            if not is_existed:
                self.pruned_rules.add(pruned_rule)

    def append(self, car, minSup, minConf):
        for item in car.rules:
            self._add(item, minSup, minConf)


def statisfies(datacase, rule):
    for item in rule.cond_set:
        if datacase[item] != rule.cond_set[item]:
            return None
    return datacase[-1] == rule.class_label


def errors_of_rule(r, dataset):
    errors_number = 0
    for case in dataset:
        if statisfies(case, r) == False:
            errors_number += 1
    return errors_number
        

def prune(rule, dataset):
    min_rule_err = sys.maxsize
    pruned_rule = rule
    def find_prune_rule(curr_rule, dataset):
        nonlocal min_rule_err
        nonlocal pruned_rule
        rule_err = errors_of_rule(curr_rule, dataset)
        if rule_err < min_rule_err:
            min_rule_err = rule_err
            pruned_rule = curr_rule
        curr_rule_cond_set = list(curr_rule.cond_set)
        if len(curr_rule_cond_set) >= 2:
            for a in curr_rule_cond_set:
                tmp_cond_set = dict(curr_rule.cond_set)
                tmp_cond_set.pop(a)
                tmp_rule = RuleItem(tmp_cond_set, curr_rule.class_label, dataset)
                tmp_rule_err = errors_of_rule(tmp_rule, dataset)
                if tmp_rule_err <= min_rule_err:
                    min_rule_err = tmp_rule_err
                    pruned_rule = tmp_rule
                    if len(tmp_cond_set) >= 2:
                        find_prune_rule(tmp_rule, dataset)

    find_prune_rule(rule, dataset)
    return pruned_rule
