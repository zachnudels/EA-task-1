import sys
sys.path.insert(0, 'evoman')
from controller import Controller

class RNNController(Controller):
    def __init__(self, ctrnn, time_const):
        self.ctrnn = ctrnn
        self.time_const = time_const


    def control(self, inputs, controller):
        output = self.ctrnn.advance(inputs, self.time_const, self.time_const)

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

