"""
This module implements production rule systems, which include
a forward chaining system and a backward chaining system.
"""

def get_triggered_rule(beliefs, rules):
    """
    Return the first rule that allows the system to
    add something new to its beliefs. If no such rule
    exists, return None.
    """
    for rule in rules:
        if (rule[0] in beliefs) and (rule[1] not in beliefs):
            return rule
    return None

def forward_chain(beliefs, rules):
    """
    Automatically perform forward chaining using a given
    set of initial beliefs and list of rules. 

    Return a tuple where the first item is the final set
    of beliefs and the second item is a list of rules that
    were triggered, in the order they were triggered.
    """
    triggered_rules = []
    current_beliefs = beliefs.copy()
    current_triggered_rule = get_triggered_rule(current_beliefs, rules)

    while current_triggered_rule != None:
        triggered_rules.append(current_triggered_rule)
        current_beliefs.add(current_triggered_rule[1])
        current_triggered_rule = get_triggered_rule(current_beliefs, rules)
    return (current_beliefs, triggered_rules)

def backward_chain(beliefs, rules, goal):
    """
    Return True if the given goal (a string) can be proven
    with the given beliefs and rules, else return False.
    """
    if goal == "" or goal in beliefs:
        return True
    new_goals = []
    new_rules = rules.copy()

    for rule in rules:
        if rule[1] == goal:
            new_goals.append(rule[0])
            new_rules.remove(rule)

    for new_goal in new_goals:
        if backward_chain(beliefs, new_rules, new_goal):
            return True
    return False
