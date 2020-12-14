
import numpy as np
import random
import torch

class DataGenerator(object):
    """
    description (fixme)
    """
    def __init__(self, datasource, batch_size, num_samples_per_class):

        self.datasource = datasource
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class

        if self.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = [0.1, 5.0]
            self.phase_range = [0, np.pi]
            self.input_range = [-5.0, 5.0]
            self.dim_input = 1
            self.dim_output = 1
            # (fixme: insert other possible datasources)
        else:
            raise ValueError('Unrecognized data source')

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        """
        Description (fixme)
        """
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], self.batch_size)
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], self.batch_size)
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])

        for i in range(self.batch_size):
            inputs[i] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class,1])
            outputs[i] = amp[i] * np.sin(inputs[i] - phase[i])

        return inputs, outputs, amp, phase
