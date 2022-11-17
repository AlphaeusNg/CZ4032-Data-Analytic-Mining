# Classification and Association Rule Mining (CAR)
# Classification Builder Algorithm
# Rule Generator (CBA-RG)
# Classifier builder (CBA-CB) - Select a subset of rules


# CBA (R,D) Algorithm: R-CARs, D-Training Data
import sys
import rg
from functools import cmp_to_key
import collections
# Retrieve an array of attributes in string format?


# Sort the CAR by their confidence and then support. Meaning if confidence of
# ri > rj, or if ri == rj, but ri's support is greater than rj, then select ri.
# Else select the rule with the higher confidence.
# This function sorts the set of generated rules CAR according to the relation ">", and returns the sorted rule list
# It stores all the rules in a list, ruleList (RL)

def sort(car):
    def compare_method(a, b):
        if a.conf < b.conf:     # 1. the confidence of ri > rj
            return 1
        elif a.conf == b.conf:
            if a.sup < b.sup:       # 2. their confidences are the same, but support of ri > rj
                return 1
            elif a.sup == b.sup:
                if len(a.cond_set) < len(b.cond_set):   # 3. both confidence & support are the same, ri earlier than rj
                    return -1
                elif len(a.cond_set) == len(b.cond_set):
                    return 0
                else:
                    return 1
            else:
                return -1
        else:
            return -1

    # Converts CAR.rules into a list
    ruleList = list(car.rules)

    # Sorts the ruleList via calling the compare method. The key to compare is equal to the rule object, and in turn it's conf and sup.
    ruleList.sort(key=cmp_to_key(compare_method))
    return ruleList

# This function checks whether the rule covers the data case (a line in table with the class label at the end of list) or not
# if covers (LHS of rule is the same as the data case, and they belong to the same class), return True;
# else if LHSs are the same while the class labels are different, return False;
# else (LHSs are different), return None
def is_satisfy(datacase, rule):
    for item in rule.cond_set:
        if datacase[item] != rule.cond_set[item]:
            return None
    if datacase[-1] == rule.class_label:
        return True
    else:
        return False

def is_satisfy_case(datacase, rule):
    for item in rule.cond_set:
        if datacase[item] != rule.cond_set[item]:
            return False
    return True

"""
Description: The implementation of a naive algorithm for CBA-CB: M1. The class Classifier includes the set of selected
    rules and defaultClass, which can be expressed as <r1, r2, ..., rn, defaultClass>. Method classifier_builder_m1
    is the main method to implement CBA-CB: M1.
Input: a set of CARs generated from rule_generator (see rg.py) and a dataset got from pre_process
    (see pre_processing.py)
Output: a classifier
Reference: http://www.docin.com/p-586554186.html
"""

class Classifier:
    """
    This class is our classifier. The ruleList and defaultClass are useful for outer code.
    """

    # ruleList contains a list of the rules... HAHA
    # defaultClass will be the rule's default Class
    # errorList will be a list of the computed errors
    # defaultClassList will contain a default of
    def __init__(self):
        self.ruleList = list()
        self.defaultClass = None
        self._errorList = list()
        self._defaultClassList = list()

    # insert a rule into ruleList, then choose a default class, and calculate the errors (see line 8, 10 & 11)
    def insert(self, rule, dataset):
        self.ruleList.append(rule)             # insert r at the end of C
        self.select_default_class(dataset)     # select a default class for the current C
        self.compute_error(dataset)            # compute the total number of errors of C

    # select the majority class in the remaining data
    def select_default_class(self, dataset):
        classColumns = [x[-1] for x in dataset]
        class_label = set(classColumns)
        max = 0
        current_defaultClass = None
        for label in class_label:
            if classColumns.count(label) >= max:
                max = classColumns.count(label)
                current_defaultClass = label
        self._defaultClassList.append(current_defaultClass)

    # compute the sum of errors
    def compute_error(self, dataset):
        if len(dataset) <= 0:
            self._errorList.append(sys.maxsize)
            return

        errorNumber = 0

        # the number of errors that have been made by all the selected rules in C
        for case in dataset:
            isCover = False
            for rule in self.ruleList:
                if is_satisfy(case, rule):
                    isCover = True
                    break
            if not isCover:
                errorNumber += 1

        # the number of errors to be made by the default class in the training set
        classColumns = [x[-1] for x in dataset]
        errorNumber += len(classColumns) - classColumns.count(self._defaultClassList[-1])
        self._errorList.append(errorNumber)

    # Discards all the other rules that are not selected
    def discard(self):
        # find the first rule p in C with the lowest total number of errors and drop all the rules after p in C
        index = self._errorList.index(min(self._errorList))
        self.ruleList = self.ruleList[:(index+1)]
        self._errorList = None

        # assign the default class associated with p to defaultClass
        self.defaultClass = self._defaultClassList[index]
        self._defaultClassList = None

    # just print out all selected rules and default class in our classifier
    def print(self):
        for rule in self.ruleList:
            rule.print_rule()
        print("defaultClass:", self.defaultClass)

# main method of CBA-CB: M1
def classifier_builder_m1(cars, dataset):
    classifier = Classifier()
    carsList = sort(cars)

    # for each rule r in RL, do
    for rule in carsList:
        temp = []
        mark = False

        # if D is not in RL and r classifies at least one tuple in D correctly
        for i in range(len(dataset)):
            isSatisfyValue = is_satisfy(dataset[i], rule)
            if isSatisfyValue is not None:
                temp.append(i)
                if isSatisfyValue:
                    mark = True

        # If r classifies at least one tuple in D correctly, add r at the end of RL
        if mark:
            tempDataset = list(dataset)
            for index in temp:
                tempDataset[index] = []
            while [] in tempDataset:
                tempDataset.remove([])
            dataset = tempDataset

            # insert contains:
                # insert r at the end of C
                # select a default class for the current C
                # compute the total number of errors of C
            classifier.insert(rule, dataset)

    # Find the first rule p in RL with the lowest total number of errors and drop all the rules after p in RL; add default class
    # But if no rules, then don't discard
    if len(classifier.ruleList) > 0:
        classifier.discard()
    if classifier.defaultClass == None:
        # if really no default then look through dataset and pick the majority class
        compare_class_list = []

        # Loop through each class column
        for i in range(len(dataset)):
            compare_class_list.append(dataset[i][-1])
        # Get total count of each prediction
        count_of_each_class = collections.Counter(compare_class_list)

        # Get the class with the majority vote
        classifier.defaultClass = max(count_of_each_class, key=count_of_each_class.get)

    return classifier

if __name__ == '__main__':
    dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
               [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    minsupport = 0.15
    minconfidence = 0.6
    cars = rg.rule_generator(dataset, minsupport, minconfidence)
    classifier = classifier_builder_m1(cars, dataset)
    classifier.print()

    print()
    dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
               [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    cars.prune_rules(dataset)
    cars.rules = cars.pruned_rules
    classifier = classifier_builder_m1(cars, dataset)
    classifier.print()


