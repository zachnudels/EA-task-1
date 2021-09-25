import sys
sys.path.insert(0, 'evoman')
from controller import Controller
import numpy as np


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
    projectile_distances = np.array([np.sqrt(inputs[i]**2 + inputs[i+1]**2) for i in range(4, 19, 2)])
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


class EngineeredRNNController(Controller):
    def __init__(self, ctrnn, time_const):
        self.ctrnn = ctrnn
        self.time_const = time_const


    def control(self, inputs, controller):
        eng_inputs = construct_features(inputs)
        output = self.ctrnn.advance(eng_inputs, self.time_const, self.time_const)

        left, right, jump, shoot, release = 0, 0, 0, 0, 0


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


