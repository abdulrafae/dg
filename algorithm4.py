import collections
from collections import UserDict
import math
import random
import sys
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--infile', '-i', type=str, help='Input file')
parser.add_argument('--outpath', '-o', type=str, help='Output path')
args = parser.parse_args()

infile = args.infile
outpath = args.outpath

class Partition(UserDict):
    def __init__(self, input_pairs: dict, relative_frequency: dict, k: int, n: int):
        self.relative_frequency = relative_frequency
        self.k = k
        self.n = n
        self.entropy = 0
        self.cluster_frequency = [0] * self.k
        super().__init__(input_pairs)

    def __setitem__(self, word, cluster):
        if word in self.data:
            self.__delitem__(word)
        word_frequency = self.relative_frequency[word]
        current_cluster_frequency = self.cluster_frequency[cluster]
        #self.entropy = self.entropy + xlogx(current_cluster_frequency) - xlogx(current_cluster_frequency + word_frequency)
        self.cluster_frequency[cluster] = current_cluster_frequency + word_frequency
        self.data[word] = cluster

    def __delitem__(self, word):
        if word in self.data:
            current_cluster = self.data[word]
            current_cluster_frequency = self.cluster_frequency[current_cluster]
            word_frequency = self.relative_frequency[word]
            #self.entropy = self.entropy + xlogx(current_cluster_frequency) - xlogx(current_cluster_frequency - word_frequency)
            self.cluster_frequency[current_cluster] = current_cluster_frequency - word_frequency
            del self.data[word]
        else:
            print('This word is not clustered')

    def lowest_frequency_cluster(self):
        lowest = 0
        for i in range(self.k):
            if self.cluster_frequency[i] < lowest:
                lowest = i
        return lowest

    def get_cluster_size(self):
        cluster_size = np.array([0] * self.k)
        for word in self.data:
            cluster_size[self.data[word]] = cluster_size[self.data[word]] + 1
        return f'The largest cluster size is {np.amax(cluster_size)}, the smallest cluster size is {np.amin(cluster_size)}'

    def remove(self, word, cluster):
        if self.data[word] == cluster:
            self.__delitem__(word)

    def swap(self, new_word, new_cluster, old_word, old_cluster):
        if self.data[old_word] == old_cluster:
            self.__delitem__(old_word)
            self.__setitem__(new_word, new_cluster)

    def include(self, word, cluster):
        if word not in self.data:
            self.__setitem__(word, cluster)


def finalize_solution(solution: Partition, vocabulary):
    lowest_frequency_cluster = solution.lowest_frequency_cluster()
    print(lowest_frequency_cluster, solution.cluster_frequency[lowest_frequency_cluster])
    for word in vocabulary:
        if word not in solution.data:
            solution.include(word, lowest_frequency_cluster)
    print(solution.cluster_frequency[lowest_frequency_cluster])
    return solution


def get_important_word(vocabulary, relative_frequency):
    entropy = -10
    important_word = vocabulary[0]
    for word in vocabulary:
        if (-1) * xlogx(relative_frequency[word]) > entropy:
            important_word = word
            entropy = (-1) * xlogx(relative_frequency[word])
    print(f'The important word is {important_word}')
    return important_word


def xlogx(x: float) -> float:
    if x > 0:
        return x * math.log2(x)
    else:
        return 0


def inclusion_update(cluster_frequency, word_frequency):
    return + xlogx(cluster_frequency) - xlogx(cluster_frequency + word_frequency)


def removal_update(cluster_frequency, word_frequency):
    return + xlogx(cluster_frequency) - xlogx(cluster_frequency - word_frequency)


def inclusion_entropy(entropy, cluster_frequency, word_frequency):
    return entropy + inclusion_update(cluster_frequency, word_frequency)


def removal_entropy(entropy, cluster_frequency, word_frequency):
    return entropy + removal_update(cluster_frequency, word_frequency)


def swap_entropy(entropy, old_cluster, old_word_frequency, old_cluster_frequency, new_cluster, new_word_frequency, new_cluster_frequency):
    if old_cluster == new_cluster:
        return entropy + inclusion_update(old_cluster_frequency, new_word_frequency - old_word_frequency)
    else:
        return entropy + removal_update(old_cluster_frequency, old_word_frequency) + inclusion_update(new_cluster_frequency, new_word_frequency)


def define_text_parameters(frequency_dictionary, text_length):
    print('We are defining text parameters')
    frequency_vector = {}
    vocabulary = set()
    for word in frequency_dictionary:
        frequency_vector[word] = frequency_dictionary[word] / text_length
        vocabulary.add(word)
    return frequency_vector, vocabulary


def get_approximate_clustering(vocabulary, epsilon, relative_frequency, n, k, existing_solution):
    important_word = get_important_word(vocabulary, relative_frequency)
    partition = Partition({important_word: 0}, relative_frequency, k, n)
    current_entropy = (-1) * xlogx(relative_frequency[important_word])
    partition_changed = True
    threshold = 1 + epsilon/(k * len(vocabulary)) ** 4
    iteration = 0

    while partition_changed:
        partition_changed = False
        shuffled_vocabulary = vocabulary
        random.shuffle(shuffled_vocabulary)
        if iteration % 2000 == 0:
            print(f'Iteration {iteration}, solution size {len(partition.data)}, entropy {current_entropy}')
        for incoming_word in shuffled_vocabulary:
            cluster_shift = np.random.randint(0, k)
            incoming_word_frequency = relative_frequency[incoming_word]
            used_pair = False
            for c in range(k):
                incoming_cluster = (c + cluster_shift) % k
                if incoming_word in existing_solution:
                    if existing_solution[incoming_word] == incoming_cluster:
                        used_pair = True
                if not used_pair:
                    change_threshold = threshold * current_entropy
                    incoming_cluster_frequency = partition.cluster_frequency[incoming_cluster]
                    if incoming_word not in partition.data:
                        expected_entropy = inclusion_entropy(current_entropy, incoming_cluster_frequency, incoming_word_frequency)
                        if change_threshold < expected_entropy:
                            partition.include(incoming_word, incoming_cluster)
                            current_entropy = expected_entropy
                            if iteration % 2000 == 0:
                                print(f'Included {incoming_word} to {incoming_cluster}, iteration {iteration}, entropy {current_entropy}')
                            partition_changed = True
                        else:
                            partition_data_shuffled = list(partition.data.keys())
                            random.shuffle(partition_data_shuffled)
                            for outcoming_word in partition_data_shuffled:
                                outcoming_cluster = partition.data[outcoming_word]
                                outcoming_word_frequency = relative_frequency[outcoming_word]
                                outcoming_cluster_frequency = partition.cluster_frequency[outcoming_cluster]
                                expected_entropy = swap_entropy(current_entropy, outcoming_cluster, outcoming_word_frequency, outcoming_cluster_frequency, incoming_cluster, incoming_word_frequency, incoming_cluster_frequency, )
                                if expected_entropy > change_threshold:
                                    partition.remove(outcoming_word, outcoming_cluster)
                                    partition.include(incoming_word, incoming_cluster)
                                    current_entropy = expected_entropy
                                    if iteration % 2000 == 0 :#or iteration > 62000:
                                        print(f'Included {incoming_word} to {incoming_cluster}, removed {outcoming_word} from {outcoming_cluster}, iteration {iteration}, entropy {current_entropy}')
                                    partition_changed = True
                                    break
                    else:
                        outcoming_word = incoming_word
                        outcoming_word_frequency = incoming_word_frequency
                        outcoming_cluster = partition.data[incoming_word]
                        outcoming_cluster_frequency = partition.cluster_frequency[outcoming_cluster]
                        if outcoming_cluster == incoming_cluster:
                            expected_entropy = removal_entropy(current_entropy, outcoming_cluster_frequency, outcoming_word_frequency)
                            if expected_entropy > change_threshold:
                                partition.remove(outcoming_word, outcoming_cluster)
                                current_entropy = expected_entropy
                                if iteration % 2000 == 0:
                                    print(f'Removed {outcoming_word} from {outcoming_cluster}, iteration {iteration}, entropy {current_entropy}')
                                partition_changed = True
                        else:
                            incoming_cluster_frequency = outcoming_cluster_frequency - outcoming_word_frequency
                            expected_entropy = swap_entropy(current_entropy, outcoming_cluster, outcoming_word_frequency, outcoming_cluster_frequency, incoming_cluster, incoming_word_frequency, incoming_cluster_frequency)
                            if expected_entropy > change_threshold:
                                partition.remove(outcoming_word, outcoming_cluster)
                                partition.include(incoming_word, incoming_cluster)
                                current_entropy = expected_entropy
                                if iteration % 2000 == 0:
                                    print(f'Included {incoming_word} to {incoming_cluster}, removed {outcoming_word} from {outcoming_cluster}, iteration {iteration}, entropy {current_entropy}')
                                partition_changed = True
                if partition_changed:
                    break
            iteration = iteration + 1
        partition.entropy = current_entropy
        iteration = iteration + 1

    return partition


def compute_the_clustering(infile, epsilon, outpath):
    vocabulary = []
    phi = {}
    text_length = 0
    with open(infile, 'r', encoding="utf-8") as vocab_file:
        for line in vocab_file:
            y = line.strip().split()
            print(y)
            vocabulary.append(y[0])
            phi[y[0]] = float(y[2])
            text_length = text_length + int(y[2])
    print(f'{infile} text length {text_length} vocabulary size {len(vocabulary)}')
    for word in vocabulary:
        phi[word] = phi[word] / text_length
    k = len(vocabulary) // 20
    print('Vocabulary and relative frequency vector are computed')
    print(f'Vocabulary size: {len(vocabulary)}')
    print(f'Number of entries in relative frequency: {len(phi)}')
    print(f'Text size: {text_length}')
    solution_1 = get_approximate_clustering(vocabulary, epsilon, phi, text_length, k, {})
    print('Solution 1 is computed')
    print(f'Entropy = {solution_1.entropy}')
    print(f'Solution 1 size: {len(solution_1.data)}')
    print(solution_1.get_cluster_size())
    existing_solution = solution_1.data
    solution_2 = get_approximate_clustering(vocabulary, epsilon, phi, text_length, k, existing_solution)
    print('Solution 2 is computed')
    print(f'Entropy = {solution_2.entropy}')
    print(f'Solution 2 size: {len(solution_2.data)}')
    print(solution_2.get_cluster_size())
    if solution_2.entropy > solution_1.entropy:
        intermediate_solution = solution_2
        print('We chose solution 2')
        print(f'Its entropy: {intermediate_solution.entropy}')
    else:
        intermediate_solution = solution_1
        print('We chose solution 1')
        print(f'Its entropy: {intermediate_solution.entropy}')
        print(f'The size of intermediate solution is: {len(intermediate_solution.data)}')
        print(f'The size of vocabulary is: {len(vocabulary)}')
    if len(intermediate_solution.data) < len(vocabulary):
        print('We need to finalize the solution')
        final_solution = finalize_solution(intermediate_solution, vocabulary)
    else:
        final_solution = intermediate_solution
        print('We do not finalize the solution')
    result = f'File {infile} has the final solution size: {len(final_solution.data)} ' + f'The final entropy: {final_solution.entropy}' + ' ' + final_solution.get_cluster_size()
    #result_file_name = infile[:-4] + "_result.txt"
    #print(result_file_name)
    #with open(result_file_name, 'w', encoding="utf-8") as result_file:
     #   result_file.write(result)
    #result_file.close()
	output_file_name = outpath+"/algo4_grouping.txt"
    with open(output_file_name, 'w', encoding="utf-8") as final_clustering:
        for current_word in final_solution.data:
            final_clustering.write(f'{current_word} : {final_solution[current_word]}\n')
	final_clustering.close()

user_epsilon = 1 #desired precision
compute_the_clustering(infile, user_epsilon)


