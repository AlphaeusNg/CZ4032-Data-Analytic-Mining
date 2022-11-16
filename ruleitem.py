class RuleItem:
    def __init__(self, cond_set, class_label, dataset):
        self.cond_set = cond_set
        self.class_label = class_label
        self.cond_sup_count, self.rule_sup_count = self.getSupCount(dataset)
        self.sup = self.getSup(len(dataset))
        self.conf = self.getConf()

    def getSupCount(self, dataset):
        cond_sup_count = 0
        rule_sup_count = 0
        for case in dataset:
            for i in self.cond_set:
                if self.cond_set[i] != case[i]:
                    return cond_sup_count, rule_sup_count 
            cond_sup_count += 1
            if self.class_label == case[-1]:
                rule_sup_count += 1
        return cond_sup_count, rule_sup_count

    def getSup(self, dataset_size):
        return self.rule_sup_count / dataset_size

    def getConf(self):
        if self.cond_sup_count != 0:
            return self.rule_sup_count / self.cond_sup_count
        else:
            return 0

    def print(self):
        cond_set_output = ''
        for item in self.cond_set:
            cond_set_output += '(' + str(item) + ', ' + str(self.cond_set[item]) + '), '
        cond_set_output = cond_set_output[:-2]
        print('<({' + cond_set_output + '}, ' + str(self.cond_sup_count) + '), (' +
              '(class, ' + str(self.class_label) + '), ' + str(self.rule_sup_count) + ')>')

    def print_rule(self):
        cond_set_output = ''
        for item in self.cond_set:
            cond_set_output += '(' + str(item) + ', ' + str(self.cond_set[item]) + '), '
        cond_set_output = '{' + cond_set_output[:-2] + '}'
        print(cond_set_output + ' -> (class, ' + str(self.class_label) + ')')
