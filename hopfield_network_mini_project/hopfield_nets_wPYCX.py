import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage.filters import threshold_otsu

import networkx as nx
import numpy as np

import pycxsimulator
import matplotlib
matplotlib.use('TkAgg')

# converts png image to n by n binary matrix
def digit_matrix(digit, n=10):
    image_path = 'hopfield_patterns/digits/{}.png'.format(digit)
    img = mpimg.imread(image_path)

    if img.ndim == 3:
        gray_image = np.mean(img, axis=2)
    else:
        gray_image = img

    def convert_to_binary_matrix(image, n):
        resized_image = resize(image, (n, n), anti_aliasing=True)

        threshold = threshold_otsu(resized_image)
        binary_image = np.where(resized_image < threshold, -1, 1)

        return binary_image

    binary_image_matrix = convert_to_binary_matrix(gray_image, n)

    # plt.imshow(binary_image_matrix, cmap='binary', interpolation='nearest')
    # plt.show()
    # print(binary_image_matrix)

    return binary_image_matrix


# imprint patterns
def imprint_patterns(G, pattern_list):
    num_patterns = len(pattern_list)
    num_nodes = G.number_of_nodes()

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            weight = sum(pattern_list[k][i] * pattern_list[k][j]
                         for k in range(num_patterns))
            G[i][j]['weight'] = weight
            G[j][i]['weight'] = weight


# generate random pattern
def generate_random_pattern(n):
    return np.random.choice([-1, 1], size=n)


# generate n digit patterns
def generate_digit_patterns(n, dim):
    pattern_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    pats = []
    for pat in pattern_list[0:n]:
        pats.append(digit_matrix(pat, dim).flatten())

    return pats

# # generate n random patterns
# def generate_random_patterns(n, dim):
#     pattern_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#     pats = []
#     for pat in pattern_list[0:n]:
#         pats.append(np.random.choice([-1, 1], size=dim*dim))

#     return pats


# compute for the max similarity percentage in a patterns group
def compute_error_percentage(states, initial_patterns):
    percetages = []

    for pattern in initial_patterns:
        percetages.append(100 - (sum(states.astype(int) == pattern.flatten())/states.size)*100)
        pattern = pattern*(-1)
        percetages.append(100 - (sum(states.astype(int) == pattern.flatten())/states.size)*100)

    return min(percetages)


# add noise to matrix
def add_noise(matrix, noise_level):
    noisy_matrix = matrix.copy()

    num_elements = matrix.size
    num_flips = int(noise_level * num_elements)

    flip_indices = np.random.choice(num_elements, num_flips, replace=False)
    row_indices, col_indices = np.unravel_index(flip_indices, matrix.shape)

    noisy_matrix[row_indices, col_indices] = np.random.choice([1, -1], size=num_flips, replace=True)

    return noisy_matrix


# computes for hopfield network
def compute_energy(states):
    global hop_network
    energy = 0
    for i in hop_network.nodes():
        for j in hop_network.neighbors(i):
            energy -= 0.5 * hop_network[i][j]['weight'] * states[i] * states[j]
    return energy

# prints image
def show_image(states):
    img = np.reshape(states, (dimension, dimension))
    plt.imshow(img, cmap='binary', interpolation='nearest')
    plt.show()


global hop_network, patterns, states, dimension, num_nodes, num_patterns, noise_level
dimension = 10
num_patterns = 1
noise_level = .1
initial_pattern = 0
energy_values = []


# pycx simulation parameters
def change_dimension(val=dimension):
    global dimension, num_nodes
    dimension = int(val)
    num_nodes = dimension*dimension
    return int(val)


def change_num_patterns(val=num_patterns):
    global patterns, num_patterns, dimension
    num_patterns = int(val)
    patterns = generate_digit_patterns(num_patterns, dimension)
    return int(val)


def change_noise_level(val=noise_level):
    global noise_level
    noise_level = float(val)
    return float(val)


def change_initial_pattern(val=initial_pattern):
    global initial_pattern
    initial_pattern = int(val)
    return int(val)


# simulation functions (initialize, update, observe)
def initialize():
    global hop_network, patterns, states, num_nodes, num_patterns

    hop_network = nx.complete_graph(num_nodes)
    for (i, j) in hop_network.edges():
        hop_network[i][j]['weight'] = 0

    imprint_patterns(hop_network, patterns)
    # states = np.random.choice([-1, 1], size=num_nodes)
    states = add_noise(digit_matrix(
        initial_pattern, dimension), noise_level).flatten()


def observe():
    global states, dimension, num_nodes, num_patterns

    img = np.reshape(states, (dimension, dimension))
    plt.clf()
    plt.imshow(img, cmap='binary', interpolation='nearest')
    plt.show()


def update():
    global hop_network, patterns, states, dimension, num_nodes, num_patterns

    # # SYNCHRONOUS UPDATE
    # new_states = np.zeros(len(states))
    # for i in hop_network.nodes():
    #     net_input = sum(hop_network[i][j]['weight'] * states[j]
    #                     for j in hop_network.neighbors(i))
    #     new_states[i] = int(np.sign(net_input))
    #     if new_states[i] == 0:
    #         new_states[i] = 1

    # states = new_states.copy()

    # # ASYNCHRONOUS RANDOM UPDATE
    i = np.random.choice(hop_network.nodes())
    net_input = sum(hop_network[i][j]['weight'] * states[j]
                    for j in hop_network.neighbors(i))
    states[i] = int(np.sign(net_input))
    if states[i] == 0:
        states[i] = 1  # In case of 0, we assume a state of 1


pycxsimulator.GUI(parameterSetters=[change_dimension, change_num_patterns, change_noise_level, change_initial_pattern]).start(
    func=[initialize, observe, update])
