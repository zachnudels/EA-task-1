import sys

sys.path.insert(0, 'evoman')

import numpy as np
from demo_controller import player_controller


def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))


class EngineeredController(player_controller):
    def __init__(self, _n_hidden):
        super().__init__(_n_hidden)

    def control(self, inputs, controller):
        inputs = np.array(construct_features(inputs))

        if self.n_hidden[0] > 0:
            # Preparing the weights and biases from the controller of layer 1

            # Biases for the n hidden neurons
            bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            # Weights for the connections from the inputs to the hidden nodes
            weights1_slice = len(inputs) * self.n_hidden[0] + self.n_hidden[0]
            weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs), self.n_hidden[0]))

            # Outputs activation first layer.
            output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

            # Preparing the weights and biases from the controller of layer 2
            bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
            weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

            # Outputting activated second layer. Each entry in the output is an action
            output = sigmoid_activation(output1.dot(weights2) + bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))

            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # takes decisions about sprite actions
        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]


def g(x):
    return np.power(2, -(np.square(x / 150)))


def transform_distance(x):
    if x == 0:
        return 0
    elif x > 0:
        return g(x)
    else:
        return -1 * g(x)


def is_player_facing_enemy(inputs):
    return (inputs[0] < 0 and inputs[2] > 0) or (inputs[0] > 0 and inputs[2] < 0)


def remove_three(inputs):
    projectile_distances = np.array([np.sqrt(inputs[i] ** 2 + inputs[i + 1] ** 2) for i in range(4, 19, 2)])
    projectile_distances.sort()
    return projectile_distances[:-3]


def construct_features(inputs):
    en_feats = [transform_distance(inputs[0]),
                transform_distance(inputs[1]),
                is_player_facing_enemy(inputs),
                inputs[3]
                ]
    for dist in remove_three(inputs):
        en_feats.append(transform_distance(dist))

    return en_feats
