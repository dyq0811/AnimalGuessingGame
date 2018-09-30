"""
This module implements a simplified ACT-R learner
to recognize the past-tense of English verbs.
"""

import numpy as np

def baseline_activation(d, lags):
    """
    Take two parameters: a float d and a list of
    lags. Return the baseline activation for a chunk
    with the given lag times.
    """
    total_sum = 0
    for lag in lags:
        total_sum += lag ** (-d)
    if total_sum == 0:
        return float("-inf")
    return np.log(total_sum)

def compute_all_baselines(file_path):
    """
    Take one file path as a parameter: a file of time
    lags with each column corresponding to one verb.
    Return a list of all baseline activations.
    """
    lags_cols = read_lags_cols(file_path)
    activations = []
    for col in lags_cols:
        activations.append(baseline_activation(0.5, col))
    return activations

def read_lags_cols(file_path):
    """
    Read and transform the file of lag times into columns.
    Return a list of lists, where each element is a column.
    """
    number_of_words = len(open(file_path).readline().split(","))
    file = open(file_path)
    lags_cols = [[] for i in range(number_of_words)]
    for line in file:
        data = line.split(",")
        for i in range(number_of_words):
            lags_cols[i].append(float(data[i].strip()))
    return lags_cols

def read_frequency(file_path):
    """
    Read the file of frequencies. Return a dictionary whose
    keys are verbs and values are corresponding frequencies.
    """
    file = open(file_path)
    next(file)
    word_frequency_map = {}
    for line in file:
        word, frequency = line.split(",")
        word_frequency_map[word] = float(frequency.strip())
    return word_frequency_map

def read_verb_chunks(file_path):
    """
    Read the file of verb chunks. Return a dictionary whose
    keys are verbs and values are lists that contain the
    corresponding stem and suffix.
    """
    file = open(file_path)
    next(file)
    word_stem_suffix_map = {}
    for line in file:
        verb, stem, suffix = line.split(",")
        word_stem_suffix_map[verb] = [stem, suffix.strip()]
    return  word_stem_suffix_map

def retrieval_time(activation, F, f):
    """
    Take three parameters: the activation, F, and f.
    Return the retrieval time for the chunk.
    """
    return F * np.exp(-f * activation)

def regular_utility():
    """
    Compute the regular utility using Ui = Pi⋅G − Ci, where
    Pi = 1, G = 5, Ci = 1.2.
    """
    return 1 * 5 - 1.2

def retrieval_utility(frequency_path, lags_path):
    """
    Compute the retrieval utility using Ui = Pi⋅G − Ci.
    Pi is computed as the ratio of the total frequency of words
    above the activation threshold to the total frequency of
    all words. Ci is equal to the average retrieval time, which
    is the average of the retrieval times for all words in the
    vocabulary that can be retrieved, weighted by their frequency.
    """
    baseline_activations = compute_all_baselines(lags_path)
    word_frequency_map = read_frequency(frequency_path)
    verbs = list(word_frequency_map.keys())
    frequencies = list(word_frequency_map.values())

    retrievable_verbs = {}
    for i in range(len(baseline_activations)):
        if baseline_activations[i] > 0.3:
            retrievable_verbs[verbs[i]] = i
    prob = sum([word_frequency_map[v] for v in retrievable_verbs])/sum(frequencies)

    retrieval_times = [retrieval_time(A, 0.5, 0.25) for A in baseline_activations]
    weighted_times = [(frequencies[i] * retrieval_times[i]) for i in retrievable_verbs.values()]
    cost = sum(weighted_times)/sum([frequencies[i] for i in retrievable_verbs.values()])
    return prob * 5 - cost
