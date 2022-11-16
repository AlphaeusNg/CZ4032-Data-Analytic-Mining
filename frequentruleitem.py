class FrequentRuleItems:
    def __init__(self):
        self.freq_ruleitems_set = set()

    def getSize(self):
        return len(self.freq_ruleitems_set)

    def add(self, rule_item):
        is_existed = False
        for item in self.freq_ruleitems_set:
            if item.class_label == rule_item.class_label and item.cond_set == rule_item.cond_set:
                is_existed = True
                break
        if not is_existed:
            self.freq_ruleitems_set.add(rule_item)

    def append(self, sets):
        for item in sets.frequent_ruleitems:
            self.add(item)

    def print(self):
        for item in self.freq_ruleitems_set:
            item.print()