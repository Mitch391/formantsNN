import tensorflow as tf
import settings
import os
import numpy as np
import sound_to_spectrogram
import random

class SpectrogramTxtDataGen(tf.keras.utils.Sequence):
    # x_set: path to input items
    # y_set: output items
    def __init__(self, x_set=None, y_set=None, batch_size=settings.batch_size):
        x_set=np.array([str(y) + '.txt' for y in sorted([int(x[:-4]) for x in os.listdir(settings.spectrograms_txt_dest)])])
        y_set=open(settings.straight_formants_file_src + 'formants.txt', 'r').readlines()
        new_y = {}
        for i in y_set:
            key_and_value = i.split(',')
            key = int(key_and_value[0])
            value = key_and_value[1:]
            new_y[key] = value

        self.x = x_set
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
            ys.append(self.get_y(i))

        return np.array(xs), np.array(ys)

    
    def __get_x(self, index):
        x = np.loadtxt(settings.spectrograms_txt_dest + self.x[index], delimiter='\t')
        sample = random.randint(1,10)
        x = sound_to_spectrogram.convert_spectrogram_txt(x, sample, sample+settings.spectrogram_txt_window_size)
        x = np.rot90(x)
        return x

    
    def get_y(self, index):
        y = self.y[index][:-1]
        y = np.array([float(x) for x in y])
        return y