import tensorflow as tf
import settings
import os
import numpy as np
import random
import soundfile as sf

class SoundDataGen(tf.keras.utils.Sequence):
    def __init__(self, x_set=None, y_set=None, batch_size=settings.batch_size):
        x_set = np.array(sorted([int(x[:-4]) for x in os.listdir(settings.sound_files_src)]))
        y_set = open(settings.straight_formants_file_src + 'formants.txt', 'r').readlines()

        new_x = {}
        for i in x_set:
            key = i
            value = str(i) + '.wav'
            new_x[key] = value

        new_y = {}
        for i in y_set:
            key_and_value = i.split(',')
            key = int(key_and_value[0])
            value = key_and_value[1:]
            new_y[key] = value

        self.x = new_x
        self.y = new_y
        self.batch_size = batch_size


    # length of batch
    def __len__(self):
        length = int(settings.number_of_training_items/settings.batch_size) 
        return length
    

    # get batch at index
    def __getitem__(self, index):
        xs = []
        ys = []
        for i in range(index, self.batch_size):
            xs.append(self.__get_x(i))
            ys.append(self.__get_y(i))

        return np.array(xs), np.array(ys)

    def __get_x(self, index):
        x = sf.SoundFile(settings.sound_files_src + self.x[index], 'r').read()
        start = random.randint(0, settings.sound_sample_size)
        data = get_005s(x, start)
        return data

    
    def __get_y(self, index):
        y = self.y[index][:-1]
        y = np.array([float(x) for x in y])
        return y

def get_005s(sound, start=0):
    return np.array(sound[start:start+settings.sound_sample_size])