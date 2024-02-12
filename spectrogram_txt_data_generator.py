import tensorflow as tf
import settings
import os
import numpy as np
import sound_to_spectrogram

class SpectrogramTxtDataGen(tf.keras.utils.Sequence):
    # x_set: path to input items
    # y_set: output items
    def __init__(self,
                 x_set=np.array([str(y) + '.txt' for y in sorted([int(x[:-4]) for x in os.listdir(settings.spectrograms_text_dest)])]),
                 y_set=open(settings.straight_formants_file_src + 'formants.txt', 'r').readlines(),
                 batch_size=settings.batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = int(batch_size)
    

    # length of batch
    def __len__(self):
        return int(settings.number_of_training_items/settings.batch_size)
    
    # get batch at index
    def __getitem__(self, index):
        xs = []
        ys = []
        for i in range(index, self.batch_size):
            xs.append(self.__get_x(i))
            ys.append(self.get_y(i))

        return np.array(xs), np.array(ys)

    
    def __get_x(self, index):
        x = np.loadtxt(settings.spectrograms_text_dest + self.x[index], delimiter='\t')
        x = sound_to_spectrogram.convert_spectrogram_txt(x)
        return x

    
    def get_y(self, index):
        y = self.y[index].split(',')[1:-1]
        y = np.array([float(x) for x in y])
        return y