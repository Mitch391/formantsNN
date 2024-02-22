import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sound_data
import scipy
import sound_to_spectrogram
import settings
import cv2
import generate_training_data
from generators import spectrogram_txt_data_generator
from generators import sound_generator

sample_rate = settings.sample_rate

def get_training_data(training_files, results_file):
    sounds = []
    results = []
    for sound_file in training_files:
        sound = sound_data.Sound(sound_file)
        number = sound_file.split("/")[-1]
        number = int(number[:-4])
        sounds.append([number, sound])
    sounds = sorted(sounds)

    for line in results_file:
        results.append([float(x) for x in line[:-1].split(",")[:-1]])
    sounds = np.array(sounds)
    results = np.array(results)
    return sounds, results

def get_first_005s(sounds):
    samples = int(settings.sample_rate*0.05)
    for key in range(len(sounds)):
        sounds[key][1].data = sounds[key][1].data[:samples]


def flatten(sounds):
    for key in range(len(sounds)):
        sound = sounds[key]
        data = sound[1].data
        min_amp = min(data)
        max_amp = max(data)+abs(min_amp)
        sounds[key][1].data = np.array([(x+abs(min_amp))/max_amp for x in data])

def save_data(sounds):
    if os.path.isfile("./data/sounds.txt"): os.remove("./data/sounds.txt")
    file = open("./data/sounds.txt", "a")
    for s in sounds:
        file.write(' '.join(str(x) for x in s) + '\n')


def get_sounds_data_from_file(file="./data/sounds.txt"):
    sounds = []
    file = open(file)
    for f in file:
        sounds.append([float(x) for x in f[:-1].split()])
    sounds = np.array(sounds)
    return sounds

    

def get_results_data_from_file(results_file=settings.straight_formants_file_src + "formants.txt"):
    results = []
    results_file = open(results_file)
    for line in results_file:
        results.append([float(x) for x in line[:-1].split(",")[1:-1]])
    results = np.array(results)
    return results

def get_data_from_file(file="./data/sounds.txt", results_file="./data/formants/formants.txt"):
    sounds = []
    results = []
    results_file = open(results_file)
    for line in results_file:
        results.append([float(x) for x in line[:-1].split(",")[:-1]])
    file = open(file)
    for f in file:
        sounds.append([float(x) for x in f[:-1].split()])
    results = np.array(results)
    sounds = np.array(sounds)
    return sounds, results


def train_on_sound(sounds=None, results=None):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=settings.sound_sample_size),
        tf.keras.layers.GaussianNoise(0.5),
        tf.keras.layers.Dense(settings.sound_sample_size, activation='selu'),
        tf.keras.layers.Dense(int(settings.sound_sample_size/2)),
        tf.keras.layers.Dense(5)
    ])  
    model.compile(
                optimizer=tf.keras.optimizers.AdamW(),
                loss = tf.keras.losses.MeanAbsoluteError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    model.fit(sound_generator.SoundDataGen(), epochs=settings.epochs, validation_data=sound_generator.SoundDataGen())
    return model


def train_on_spectrogram_txt(sounds=None, results=None):
    # new_sounds = []
    # for s in sounds:
    #     new_sounds.append(np.rot90(s))
    # sounds = np.array(new_sounds)
    model = tf.keras.Sequential([
        # tf.keras.layers.Conv1D(116, 3, activation='relu', input_shape=(settings.spectrogram_txt_window_size, 116)),
        tf.keras.layers.Flatten(input_shape=(settings.spectrogram_txt_window_size, 116)),
        # tf.keras.layers.GaussianNoise(0.2),
        tf.keras.layers.Normalization(),
        tf.keras.layers.GaussianNoise(0.2),
        # tf.keras.layers.GaussianNoise(1.0),
        tf.keras.layers.Dense(166*settings.spectrogram_txt_window_size, activation='tanh'),
        tf.keras.layers.Dense(int((166*settings.spectrogram_txt_window_size)/2)),
        tf.keras.layers.Dense(5)
    ])  

    #     ### ~200 rsme after 500 epochs
    #     # tf.keras.layers.Dense(958, activation='relu'),
    #     # tf.keras.layers.Dropout(0.2),
    #     # tf.keras.layers.Dense(128),
    #     # tf.keras.layers.Dropout(0.2),
    #     # tf.keras.layers.Dense(5)
    # ])  
    model.compile(
                # optimizer=tf.keras.optimizers.legacy.Adam(),
                optimizer=tf.keras.optimizers.AdamW(),

                # loss = tf.keras.losses.MeanSquaredError(),
                # metrics=[tf.keras.metrics.RootMeanSquaredError()])

                loss = tf.keras.losses.MeanAbsoluteError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
                #   metrics=[tf.keras.metrics.MeanAbsoluteError()])
                #   metrics=['accuracy'])
    # model.fit(sounds, results, epochs=settings.epochs, validation_split=0.2, verbose=2)
    # model.fit(spectrogram_txt_data_generator.SpectrogramTxtDataGen(), epochs=settings.epochs, verbose=1)
    model.fit(spectrogram_txt_data_generator.SpectrogramTxtDataGen(), epochs=settings.epochs, validation_data=spectrogram_txt_data_generator.SpectrogramTxtDataGen())
    return model


def visualize_txt(spectrogram, data):
    new_spectrogram = []
    margin_y = int(0.1*len(spectrogram))
    margin_x = int(0.1*len(spectrogram[0]))
    for s in spectrogram[margin_y:-margin_y]:
        s = [255-x for x in s]
        s = s[margin_x:-margin_x]
        new_spectrogram.append(np.array(s))
    spectrogram = np.array(new_spectrogram)
    new_y = 5000
    new_x = 5000/(len(spectrogram))*len(spectrogram[0])
    implot = plt.imshow(spectrogram, extent=(0,new_x,0,new_y))
    x_step = int(new_x / len(data))
    xs = [i for i in range(0, int(new_x), x_step)]
    for key, d in enumerate(data):
        x = [xs[key] for _ in d]
        # d = [958-(i/5000)*958 for i in d]
        plt.scatter(x, d, c='r')
    plt.show()


def test_spectrogram_txt(model):
    spectrogram_image = cv2.imread(settings.test_file + '.png', flags=cv2.IMREAD_GRAYSCALE)
    spectrogram_windows = []
    spectrogram_txt = np.loadtxt(settings.test_file + '.txt', delimiter='\t')
    for i in range(0, len(spectrogram_txt[0]) - settings.spectrogram_txt_window_size, int(0.05*len(spectrogram_txt[0]))):
        x = sound_to_spectrogram.convert_spectrogram_txt(spectrogram_txt, i, i+settings.spectrogram_txt_window_size)
        spectrogram_windows.append(np.rot90(x))
        # spectrogram_windows.append(x)
    spectrogram_windows = np.array(spectrogram_windows)
    # spectrogram = np.array([sound_to_spectrogram.convert_spectrogram(spectrogram_image)])
    predictions = model.predict(spectrogram_windows)
    print(predictions[0])
    visualize_txt(spectrogram_image, predictions)

def test_sound(model):
    spectrogram_image = cv2.imread(settings.test_file + '.png', flags=cv2.IMREAD_GRAYSCALE).tolist()
    sound = sound_data.Sound(settings.test_file + '.wav').data
    spectrogram_windows = []
    for i in range(0, len(sound)-settings.sound_sample_size, settings.sound_sample_size):
        x = sound_generator.get_005s(sound, i)
        spectrogram_windows.append(x)
    spectrogram_windows = np.array(spectrogram_windows)
    predictions = model.predict(spectrogram_windows)
    print(predictions[0])
    visualize_txt(spectrogram_image, predictions)


def spectrogram_txt_model():
    # results_straight = get_results_data_from_file()
    # # results_moving = get_results_data_from_file(settings.moving_formants_file_src + 'formants.txt')
    # sounds = sound_to_spectrogram.spectrogram_txt_to_data()
    # sounds = sorted(sounds)
    # sounds = np.array([np.array(x) for _, x in sounds])
    # model = train_on_spectrogram_txt(sounds, results_straight)
    model = train_on_spectrogram_txt()
    model.save('./spectrogram_txt_model.keras')
    model = tf.keras.models.load_model('./spectrogram_txt_model.keras')
    print(model.summary())
    test_spectrogram_txt(model)

def sound_model():
    # results = get_results_data_from_file()
    # sounds = get_sounds_data_from_file()
    # model = train_on_sound(sounds, results)
    # model = train_on_sound()
    # model.save('./sound_model.keras')
    model = tf.keras.models.load_model('./sound_model.keras')
    print(model.summary())
    test_sound(model)

def main():
    # generate_training_data.generate_training_data()
    # if not os.path.isfile("./data/sounds.txt"):
    #     training_files = ('./data/sound_files/' + x for x in os.listdir('./data/sound_files'))
    #     results_file = open(settings.straight_formants_file_src + 'formants.txt')
    #     sounds, results = get_training_data(training_files, results_file)
    #     get_first_005s(sounds)
    #     #flatten(sounds)
    #     sounds = np.array([x[1].data for x in sounds])
    #     save_data(sounds)
        # sounds, results = get_data_from_file()
    # spectrogram_png_model()     #rmse: 439 after 500 epochs, 55ms/step
    # spectrogram_txt_model()     #rmse: 209 after 500 epochs, 5ms/step > 163 after 5000 epochs
    sound_model()
    

    # sounds = fourier_of_sounds(sounds)



if __name__ == "__main__":
    main()