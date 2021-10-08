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
    projectiles = [(np.sqrt(inputs[i] ** 2 + inputs[i + 1] ** 2), inputs[i], inputs[i + 1])
                   for i in range(4, 19, 2)]
    projectiles.sort(key=lambda x: x[0])
    rtn = []
    for projectile in projectiles[:-3]:
        rtn.append(transform_distance(projectile[1]))
        rtn.append(transform_distance(projectile[2]))
    return rtn


def construct_features(inputs):
    en_feats = [transform_distance(inputs[0]),
                transform_distance(inputs[1]),
                is_player_facing_enemy(inputs),
                inputs[3]
                ]
    for dist in remove_three(inputs):
        en_feats.append(transform_distance(dist))
    return en_feats
