import os
import sys
import random
import csv
import numpy as np
from collections import Counter

"""
Module for working with the animal-feature dataset.
"""

def load_name_data(filename):
    '''
    Loads a file with animal names or features.
    Returns the results as a list.
    Parameters:
    filename: name of the file where the animal names/features are stored
    '''
    names = []
    f = open(filename)
    for line in f:
        names.append(line.strip())
    f.close()
    return names

def load_animal_feature_data(filename):
    '''
    Loads a file with a binary matrix of animal-feature pairs.
    Each row represents one animal, and each column represents one feature.
    Returns a 2-dimensional list where each inner list is one row in the
    original matrix
    Parameters:
    filename: name of the file where the matrix is stored
    '''
    matrix = []
    f = open(filename)
    for line in f:
        features = []
        split_line = line.split()
        for item in split_line:
            features.append(int(item.strip()))
        matrix.append(features)
    f.close()
    return matrix

def play_game(animal_name_file, feature_name_file, matrix_file):
    '''
    Plays 5 guessing games with a human, with varying numbers of
    animals to guess from.
    
    Parameters:
    animal_name_file - name of the file containing names of the possible animals
    feature_name_file - name of the file containing features of the animals
    matrix_file: name of the file where the animal-feature matrix is stored
    '''
    animal_names = load_name_data(animal_name_file)
    feature_names = load_name_data(feature_name_file)
    feature_matrix = load_animal_feature_data(matrix_file)
    
    hypothesis_sizes = [2, 4, 8, 16, 32]
    hyp_size_indices = [i for i in range(len(hypothesis_sizes))]
    random.shuffle(hyp_size_indices)
    num_trials = [0] * len(hypothesis_sizes)
    for index in hyp_size_indices:
        num_trials[index] =  run(human_query, animal_names, feature_names, feature_matrix, hypothesis_sizes[index], sys.stdout)
        
    print("Total trials per game: {}".format(num_trials))
    print("Saving trial data to 'data/my_trial_data.txt'")
    save_trial_data(num_trials, "data/my_trial_data.txt")
    return num_trials

def save_trial_data(trial_list, out_file_name):
    '''Saves the numbers in trial list to the file out_file_name.
    Resulting file is comma separated. If list is 2-d, saves 
    such that each inner list is a comma separated row.
    '''
    if len(trial_list) != 0:
        f = open(out_file_name,"w")
        if isinstance(trial_list[0], int): 
            # 1-d list
            row = ",".join(str(i) for i in trial_list)
            f.write(row)
        else:
            # assume 2-d list
            for trial_set in trial_list:
                row = ",".join(str(i) for i in trial_set)
                f.write(row)
                f.write("\n")
        f.close()

def human_query(features, animals, animal_names=None, feature_names=None, feature_matrix=None, known_features=None):
    '''Prompts the player for a guess after showing them
    the input features and animals.

    Parameters:
    features - list of features of the target animal
    animals - list of possibilities for the target animal
    (remaining four parameters are included only to make the query function
    for human players and all computer players equivalent in their parameters)
    '''
    print("Features: {}".format(features))
    print("Animals: {}".format(animals))
    animal_guess = input("Your guess: ")
    while animal_guess not in animals:
        print("Invalid guess '{}' (did you make a typo?)".format(
            animal_guess))
        animal_guess = input("Your guess: ")
    return animal_guess  

def run(query_fn, animal_names, feature_names, feature_matrix, hyp_size, stream, known_features=None):
    '''Runs one instance of the animal guessing game. Returns the number of
    guesses until the player was correct.

    Parameters:
    query_fn - function to call to get the player's guess (may be human or computer player)
    animal_names - list of animal names
    feature_names - list of feature names
    feature_matrix - animal-feature matrix
    hyp_size - number of animals included as possibilities to guess
    stream - where to write output (sys.stdout writes to standard out)
    '''
    # Choose animals in the hypothesis set and the target animal
    animal_choices = random.sample(range(len(animal_names)), hyp_size)
    animals = [animal_names[i] for i in animal_choices]
    animal_index = random.choice(animal_choices)
    animal_name = animal_names[animal_index]
    
    # Make a list of all features we can show to the player
    animal_features = feature_matrix[animal_index]
    valid_features = []
    for i in range(len(animal_features)):
        if animal_features[i]:
            valid_features.append(feature_names[i])

    # Randomize choice order

    random.shuffle(animals)
    random.shuffle(valid_features)

    # Play until guessed
    guessed = False
    num_guesses = 0
    stream.write("-" * 70 + "\n")
    while not guessed:
        animal_guess = query_fn(valid_features[:2 + num_guesses], animals, animal_names, feature_names, feature_matrix, known_features)
        if animal_guess == animal_name:
            stream.write("Correct!\n")
            guessed = True
        else:
            stream.write("Sorry, try again.\n")
              
        stream.write("\n")
        num_guesses += 1

    return num_guesses

def guess(animal_choices, animal_features, animal_names, feature_names, feature_matrix):
    '''
    Use the possibilities for the animals and the features that
    have been given to the model and choose uniformly at random
    one of the choices that matches all of the listed features.

    Parameters:
    animal_choices - animal candidates for guessing
    animal_features - hint of features for guessing
    animal_names - list of animal names
    feature_names - list of feature names
    feature_matrix - animal-feature matrix
    '''
    animal_guesses = []
    for a in animal_choices:
        a_features = []
        mask = feature_matrix[animal_names.index(a)]
        for i in range(len(feature_names)):
            if mask[i] == 1:
                a_features.append(feature_names[i])
        result = all(elem in a_features for elem in animal_features)
        if result:
            animal_guesses.append(a)
    return random.choice(animal_guesses)

def computer_query(features, animals, animal_names, feature_names, feature_matrix, known_features=None):
    '''
    Prompts the computer for a guess giving the input features
    and animals. A guess is uniformly at random from those options
    that are consistent with the observed features (that are known).
    When knonw features is None, we assmue all features are known.

    Parameters:
    features - hint of features for guessing
    animals - animal candidats for guessing
    animal_names - list of animal names
    feature_names - list of feature names
    feature_matrix - animal-feature matrix
    know_features (optional) - features known by the player
    '''
    if known_features != None:
        features = list(set(known_features) & set(features))
    animal_guess = guess(animals, features, animal_names, feature_names, feature_matrix)
    return animal_guess

def model():
    '''
    A function model that has the computer play 500 iterations
    of each of the five hypothesis sizes (2, 4, 8, 16, and 32).
    '''
    animal_names = load_name_data("data/classes.txt")
    feature_names = load_name_data("data/predicates.txt")
    feature_matrix = load_animal_feature_data("data/predicate-matrix-binary.txt")
    stream = open(os.devnull, 'w')
    hypothesis_sizes = [2, 4, 8, 16, 32]
    hyp_size_indices = [i for i in range(len(hypothesis_sizes))]
    num_trials_list = []

    for i in range(500):
        random.shuffle(hyp_size_indices)
        num_trials = [0] * len(hypothesis_sizes)
        for index in hyp_size_indices:
            num_trials[index] = run(computer_query, animal_names, feature_names, feature_matrix, hypothesis_sizes[index], stream)
        num_trials_list.append(num_trials)
    save_trial_data(num_trials_list, "data/model_trial_data.txt")

def generate_known_features(n):
    """
    Generate n random features chosen from all the featrues.

    Parameter:
    n - number of features need to be generated
    """
    feature_names = load_name_data("data/predicates.txt")
    random.shuffle(feature_names)
    return feature_names[:n]

def bounded_model(n):
    """
    The model supposes the player knows only a subset of the
    feature pairings, i.e. n of the 85 features, where n is
    a parameter of the model. n random features are chosen.
    Take in the argument n and plays 500 iterations of each
    of the 5 hypothesis sizes.

    Parameter:
    n - number of know features
    """
    animal_names = load_name_data("data/classes.txt")
    feature_names = load_name_data("data/predicates.txt")
    feature_matrix = load_animal_feature_data("data/predicate-matrix-binary.txt")
    stream = open(os.devnull, 'w')
    hypothesis_sizes = [2, 4, 8, 16, 32]
    hyp_size_indices = [i for i in range(len(hypothesis_sizes))]
    num_trials_list = []
    known_features = generate_known_features(n)

    for i in range(500):
        random.shuffle(hyp_size_indices)
        num_trials = [0] * len(hypothesis_sizes)
        for index in hyp_size_indices:
            num_trials[index] = run(computer_query, animal_names, feature_names, feature_matrix, hypothesis_sizes[index], stream, known_features)
        num_trials_list.append(num_trials)
    save_trial_data(num_trials_list, "data/bounded_model_trial_data_" + str(n) + ".txt")
 

def get_most_unique_feature(target_features, feature_matrix):
    '''
    Get the most unique feature of the target animal, i.e.
    the feature that has least number of other animals that
    share the same feature.

    Complexity: O(m^2), feature_matrix n*m animals*features

    Parameters:
    target_features - all features of the target animal
    feature_matrix - animal-feature matrix
    '''
    feature_cols = np.matrix.transpose(np.array(feature_matrix.copy()))
    target_features_uniqueness = []
    for i in range(len(feature_cols)):
        target_features_uniqueness.append(Counter(feature_cols[i])[target_features[i]])
    unique_feature_index = target_features_uniqueness.index(min(target_features_uniqueness))
    return unique_feature_index, target_features[unique_feature_index]

def reduce_matrix_by_feature(feature_num, feature_index, feature_matrix):
    '''
    Reduce the feature matric by only keeping the animals with a given
    feature.

    Complexity: O(m), feature_matrix n*m animals*features

    Parameters:
    feature_num - the value (0 or 1) of the feature in the matrix
    feature_index - the column index of the feature in the matrix
    feature_matrix - animal-feature matrix
    '''
    reduced_matrix = feature_matrix.copy()
    for row in feature_matrix:
        if row[feature_index] != feature_num:
            reduced_matrix.remove(row)
    return reduced_matrix

def add_to_feature_dict(feature_num, feature_index, dict, features_names):
    '''
    Add the given feature to the feature dictionary based on
    it is positive or negative.

    Complexity: O(1), feature_matrix n*m animals*features

    Parameters:
    feature_num - the value (0 or 1) of the feature in the matrix
    feature_index - the column index of the feature in the matrix
    dict - the feature dictionary
    feature_names - list of feature names
    '''
    if feature_num == 1:
        dict["positive_features"].append(features_names[feature_index])
    else:
        dict["negative_features"].append(features_names[feature_index])

def find_simplest_rule(target_animal, animal_names, features_names, feature_matrix):
    '''
    Find the simplest conjunctive rule for identifying the given animal.
    A rule identifies an animal if it is true for the target animal and
    false for all other animals. The rule may include conjunctions of
    features or their negations. 

    Complexity: O(min(n,m)*m^2), feature_matrix n*m animals*features

    Parameters:
    target_animal - the animal to find simplest rule
    animal_names - list of animal names
    feature_names - list of feature names
    feature_matrix - animal-feature matrix
    '''
    simplest_rule_dict = {"positive_features":[], "negative_features":[]}
    target_features = feature_matrix[animal_names.index(target_animal)]
    current_feature_matrix = feature_matrix.copy()
    while len(current_feature_matrix) > 1:
        feature_index, feature_num = get_most_unique_feature(target_features, current_feature_matrix)
        add_to_feature_dict(feature_num, feature_index, simplest_rule_dict, features_names)
        current_feature_matrix = reduce_matrix_by_feature(feature_num, feature_index, current_feature_matrix)
    return simplest_rule_dict

def rule_to_animals(animal_names, features_names, feature_matrix, feature_dict):
    '''
    Find the simplest conjunctive rule for identifying the given animal.
    A rule identifies an animal if it is true for the target animal and
    false for all other animals. The rule may include conjunctions of
    features or their negations. 

    Complexity: O(min(n,m)*m^2), feature_matrix n*m animals*features

    Parameters:
    animal_names - list of animal names
    feature_names - list of feature names
    feature_matrix - animal-feature matrix
    feature_dict- the feature dictionary
    '''
    feature_cols = np.matrix.transpose(np.array(feature_matrix.copy()))
    positive_animal_indices = []
    negative_animal_indices = []

    for f in feature_dict["positive_features"]:
        target_col = feature_cols[features_names.index(f)]
        for i in range(len(target_col)):
            if target_col[i] == 1:
                positive_animal_indices.append(i)
    for f in feature_dict["negative_features"]:
        target_col = feature_cols[feature_names.index(f)]
        for i in range(len(target_col)):
            if target_col[i] == 0:
                negative_animal_indices.append(i)

    if feature_dict["positive_features"] == []:
        return [animal_names[i] for i in negative_animal_indices]
    if feature_dict["negative_features"] == []:
        return [animal_names[i] for i in positive_animal_indices]
    return [animal_names[i] for i in list(set(positive_animal_indices) & set(negative_animal_indices))]
