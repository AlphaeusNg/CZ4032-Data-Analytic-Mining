import rg
from cba_cb_m1 import classifier_builder_m1
from cba_cb_m1 import is_satisfy, is_satisfy_case
import random
from dataread import read
from pre_processing import pre_process
import timeit
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import collections

# Datasets that worked
# data_path = 'datasets/car.data'
# scheme_path = 'datasets/car.names' # label: 5
data_path = 'datasets/iris.data'
scheme_path = 'datasets/iris.names' # label: 4
# data_path = 'datasets/tic-tac-toe.data'
# scheme_path = 'datasets/tic-tac-toe.names' # label: 9
# data_path = 'datasets/glass.data'
# scheme_path = 'datasets/glass.names' # label: 10
# data_path = 'datasets/lymphography.data'
# scheme_path = 'datasets/lymphography.names' # label: 0
# data_path = 'datasets/haberman.data'
# scheme_path = 'datasets/haberman.names' # label: 3

# doesn't work
# data_path = 'datasets/abalone.data'
# scheme_path = 'datasets/abalone.names'

data, attributes, value_type = read(data_path, scheme_path)
#random.shuffle(data) # randoming the data results in weird results
dataset = pre_process(data, attributes, value_type)

# Set how many partitions you want
number_of_partitions = 10
block_size = int(len(dataset) / number_of_partitions)
split_point = [k * block_size for k in range(0, number_of_partitions)]
split_point.append(len(dataset))

cba_rg_total_runtime = 0
cba_cb_bagging_total_runtime = 0
total_car = 0
total_classifier_rule_num = 0
error_total_rate = 0
total_precision_score = 0
total_recall_score = 0
total_F1_score = 0

minSup=0.01
minConf=0.5

# calculate the error rate of the classifier on the dataset
def getErrorRate(classifier, dataset):
    size = len(dataset)
    error_number = 0
    for case in dataset:
        #print(case)
        is_satisfy_value = False
        for rule in classifier.ruleList:
            is_satisfy_value = is_satisfy(case, rule)
            if is_satisfy_value == True:
                break
            elif is_satisfy_value == False:
                error_number += 1
                break
        if is_satisfy_value == None:
            if classifier.defaultClass != case[-1]:
                error_number += 1
        #     for rule in classifier.ruleList:
        #         rule.print_rule()
        # print(is_satisfy_value)
    # print("Error Number", error_number)
    return error_number / size

def get_pred_only(classifier, dataset):
    # List of predicted labels
    pred = []
    for case in dataset:
        # print(case)
        rule_found = False
        for rule in classifier.ruleList:
            rule_found = is_satisfy_case(case, rule)
            if rule_found == True:
                pred.append(rule.class_label)
                break
        if rule_found == False:
            pred.append(classifier.defaultClass)
    #     print(rule_found)
    # print(pred)
    # print(true)
    return pred

def get_true_only(dataset):
    temp_true = []
    for row in dataset:
        temp_true.append(row[-1])
    return temp_true

def getSubSamples(training_dataset, number_of_partitions):
    # sub_block_size = int(len(training_dataset) / number_of_partitions)
    sub_split_point = [k * block_size for k in range(0, number_of_partitions)]
    sub_split_point.append(len(training_dataset)-1)
    sub_training_dataset = []

    # Generates a random bag of data, minus 1 because we want to maintain ~same size as original dataset
    for subsample in range(number_of_partitions):
        k = random.randint(0, number_of_partitions-1)
        sub_training_dataset.extend(training_dataset[sub_split_point[k]:sub_split_point[k+1]])

    return sub_training_dataset

def get_majority_predicted_class(predList):

    final_average_class_list = []
    compare_class_list = []
    num_of_col = len(predList[0])

    # Loop through each class column
    for i in range(num_of_col):

        # Loop through each model's prediction (except those that didn't have any which is taken out at the start)
        for pred_list_loop in predList:
            try:
                compare_class_list.append(pred_list_loop[i])
            except:
                continue

        # Get total count of each prediction
        count_of_each_class = collections.Counter(compare_class_list)

        # Get the class with the majority vote
        highest_count = max(count_of_each_class, key=count_of_each_class.get) # One limitation is that if there are 2 classes w same weight, max will just choose the first key

        # Update final list to reflect predicted class
        final_average_class_list.append(highest_count)
    return final_average_class_list

def get_precision_metrics(true, pred, total_precision_score):
    temp_total_precision_score = round(precision_score(true, pred, average='macro', zero_division=0), 3)
    # print("\nPrecision_score:", temp_total_precision_score)
    return total_precision_score + temp_total_precision_score

def get_total_recall_metrics(true, pred, total_recall_score):
    temp_total_recall_score = round(recall_score(true, pred, average='macro', zero_division=0), 3)
    # print("Recall_score:", temp_total_recall_score)
    return total_recall_score + temp_total_recall_score

def get_total_F1_score(true, pred, total_F1_score):
    temp_total_F1_score = round(f1_score(true, pred, average='macro', zero_division=0), 3)
    # print("F1_score:", temp_total_F1_score)
    return total_F1_score + temp_total_F1_score

for k in range(len(split_point)-1):
    print("\nRound %d:" % k)
    training_dataset = dataset[:split_point[k]] + dataset[split_point[k+1]:]
    test_dataset = dataset[split_point[k]:split_point[k+1]]

    # Find out Rule Generation Timings
    start = timeit.default_timer()
    cars = rg.rule_generator(training_dataset, minSup, minConf)
    # cars.prune_rules(training_dataset)
    # cars.rules = cars.pruned_rules
    end = timeit.default_timer()
    cba_rg_runtime = end - start
    cba_rg_total_runtime += cba_rg_runtime
    total_car += len(cars.rules)
    print("CARs:")
    cars.print_rule()
    # print("CBA-RG's run time with pruning: %f s" % cba_rg_runtime)
    # print("No. of CARs with pruning: %d" % len(cars.rules))

    if len(cars.rules) > 0: # in the case no CARs are gotten
        start = timeit.default_timer()
        pred_class = []

        # Build 9 Models
        for i in range(number_of_partitions-1):

            # Split dataset randomly for bagging
            subSample = getSubSamples(training_dataset, number_of_partitions-1)

            # Train the model on this subsample
            classifier = classifier_builder_m1(cars, subSample)
            print("\nSub-Classifier", str(i) + ":")
            classifier.print()
            error_rate = getErrorRate(classifier, test_dataset)
            error_total_rate += error_rate

            # Check prediction for this model
            pred_class.append(get_pred_only(classifier, test_dataset))

            total_classifier_rule_num += len(classifier.ruleList)
            # print("CBA-CB M1's run time with pruning: %f s" % cba_cb_runtime)

        # Find the majority class guess from pred_class list,
        pred = get_majority_predicted_class(pred_class)

        # Takes in the classifier, test_dataset and col of the label in the test dataset
        # (Need to modify value of 'col of the label' for each dataset)
        true = get_true_only(test_dataset)

        end = timeit.default_timer()
        cba_cb_bagging_runtime = end - start
        cba_cb_bagging_total_runtime += cba_cb_bagging_runtime

        # Update current metrics
        error_total_rate = error_total_rate/number_of_partitions
        total_classifier_rule_num = total_classifier_rule_num/ (number_of_partitions-1)
        total_precision_score = get_precision_metrics(true, pred, total_precision_score)
        total_recall_score = get_total_recall_metrics(true, pred, total_recall_score)
        total_F1_score = get_total_F1_score(true, pred, total_F1_score)



print("\nAverage CBA-RG's run time without pruning: %f ms" % ((cba_rg_total_runtime / number_of_partitions)*1000))
print("Average CBA-CB M1's run time without pruning: %f ms" % ((cba_cb_bagging_total_runtime / number_of_partitions)*1000))
print("Average Total CBA-CB M1 + RG's run time without pruning: %f ms" % (((cba_cb_bagging_total_runtime+cba_rg_total_runtime)/number_of_partitions)*1000))
print("Average CBA's error rate without pruning: %f%%" % (error_total_rate / number_of_partitions * 100))
print("Average No. of rules in classifier without pruning: %d" % int(total_classifier_rule_num / number_of_partitions))
print("Average No. of CARs without pruning: %d" % int(total_car / number_of_partitions))

print("\nPrecision_score:", round((total_precision_score/number_of_partitions), 3))
print("Recall_score:", round((total_recall_score/number_of_partitions), 3))
print("F1_score:", round((total_F1_score/number_of_partitions), 3))

