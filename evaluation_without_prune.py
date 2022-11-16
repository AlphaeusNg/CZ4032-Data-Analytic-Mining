import rg
from cba_cb_m1 import classifier_builder_m1
from cba_cb_m1 import is_satisfy
import random
from dataread import read
from pre_processing import pre_process
import timeit
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score



# Datasets that didn't work
# data_path = 'datasets/wdbc.data'
# scheme_path = 'datasets/wdbc.names'
# data_path = 'datasets/abalone.data'
# scheme_path = 'datasets/abalone.names'


# Datasets that worked
# data_path = 'datasets/car.data'
# scheme_path = 'datasets/car.names'
# data_path = 'datasets/iris.data'
# scheme_path = 'datasets/iris.names'
# data_path = 'datasets/tic-tac-toe.data'
# scheme_path = 'datasets/tic-tac-toe.names'
# data_path = 'datasets/tic-tac-toe.data'
# scheme_path = 'datasets/tic-tac-toe.names'
# data_path = 'datasets/glass.data'
# scheme_path = 'datasets/glass.names'
# data_path = 'datasets/lymphography.data'
# scheme_path = 'datasets/lymphography.names'
data_path = 'datasets/haberman.data'
scheme_path = 'datasets/haberman.names'



data, attributes, value_type = read(data_path, scheme_path)
#random.shuffle(data) # randoming the data results in weird results
dataset = pre_process(data, attributes, value_type)

block_size = int(len(dataset) / 10)
split_point = [k * block_size for k in range(0, 10)]
split_point.append(len(dataset))

cba_rg_total_runtime = 0
cba_cb_total_runtime = 0
total_car = 0
total_classifier_rule_num = 0
error_total_rate = 0

# scikit learn metrics
true = []
pred = []

minSup=0.01
minConf=0.5

# calculate the error rate of the classifier on the dataset
def getErrorRate(classifier, dataset):
    size = len(dataset)
    error_number = 0
    for case in dataset:
        is_satisfy_value = False
        for rule in classifier.ruleList:
            is_satisfy_value = is_satisfy(case, rule)
            if is_satisfy_value == True:
                break
        if is_satisfy_value == False:
            if classifier.defaultClass != case[-1]:
                error_number += 1
    return error_number / size

def getTrueAndPred(classifier, dataset, label_col):
    for rule in classifier.ruleList:
        for item in rule.cond_set:
            for case in dataset:
                if rule.cond_set[item] in case:
                    pred.append(rule.class_label)
                    true.append(case[label_col])    
    # print(pred)
    # print(true)

for k in range(len(split_point)-1):
    print("\nRound %d:" % k)
    training_dataset = dataset[:split_point[k]] + dataset[split_point[k+1]:]
    test_dataset = dataset[split_point[k]:split_point[k+1]]

    start = timeit.default_timer()
    cars = rg.rule_generator(training_dataset, minSup, minConf)
    end = timeit.default_timer()
    cba_rg_runtime = end - start
    cba_rg_total_runtime += cba_rg_runtime
    total_car += len(cars.rules)
    print("CARs:")
    cars.print_rule()
    # print("CBA-RG's run time without pruning: %f s" % cba_rg_runtime)
    # print("No. of CARs without pruning: %d" % len(cars.rules))

    if len(cars.rules) > 0: # in the case no CARs are gotten
        start = timeit.default_timer()
        classifier = classifier_builder_m1(cars, dataset)
        end = timeit.default_timer()
        cba_cb_runtime =  end - start
        cba_cb_total_runtime += cba_cb_runtime
        print("\nClassifier:")
        classifier.print()

        error_rate = getErrorRate(classifier, test_dataset)

        # Takes in the classifer, test_dataset and col of the label in the test dataset
        getTrueAndPred(classifier, test_dataset, 3)

        error_total_rate += error_rate

        total_classifier_rule_num += len(classifier.ruleList)
        # print("CBA-CB M1's run time without pruning: %f s" % cba_cb_runtime)

print("\nAverage CBA-RG's run time without pruning: %f ms" % ((cba_rg_total_runtime / 10)*1000))
print("Average CBA-CB M1's run time without pruning: %f ms" % ((cba_cb_total_runtime / 10)*1000))
print("Average Total CBA-CB M1 + RG's run time without pruning: %f ms" % (((cba_cb_total_runtime+cba_rg_total_runtime)/10)*1000))
print("Average CBA's error rate without pruning: %f%%" % (error_total_rate / 10 * 100))
print("Average No. of rules in classifier without pruning: %d" % int(total_classifier_rule_num / 10))
print("Average No. of CARs without pruning: %d" % int(total_car / 10))

# print("Predicted Labels")
# print(pred)
# print("True Labels")
# print(true)

print("\nPrecision_score: %f" % round(precision_score(true, pred, average='macro', zero_division=0),3))
print("Recall_score: %f" % round(recall_score(true, pred, average='macro', zero_division=0),3))
print("F1_score: %f" % round(f1_score(true, pred, average='macro', zero_division=0),3))
