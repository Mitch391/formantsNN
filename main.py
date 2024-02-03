import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sound_data
import scipy
import sound_to_spectrogram
import settings
import sys
import subprocess
import cv2
import generate_training_data

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

    

def get_results_data_from_file(results_file="./data/formants/formants.txt"):
    results = []
    results_file = open(results_file)
    for line in results_file:
        results.append([float(x) for x in line[:-1].split(",")[:-1]])
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


def fourier_of_sounds(sounds):
    new_sounds = []
    for s in sounds:
        new_s = scipy.fft.fft(s)
        new_sounds.append(2.0/2205 * np.abs(new_s[0:2205//2]))
    return np.array(new_sounds)


def train_on_sound(sounds, results):

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(sounds)

    model = tf.keras.Sequential([
        normalizer, 
        tf.keras.layers.Dense(2205, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(5)
    ])  
    model.compile(optimizer='adam',
                # loss=tf.reduce_mean(tf.square(expected - net)),
                loss = tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    model.fit(sounds, results, epochs=settings.epochs)
    # print(model.predict(sounds[:5]))
    return model


def train_on_spectrogram_txt(sounds, results):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(116,settings.spectrogram_txt_window_size)),
        tf.keras.layers.Dense(166*settings.spectrogram_txt_window_size, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(int((166*settings.spectrogram_txt_window_size)/2)),
        tf.keras.layers.Dense(5)

        ### ~200 rsme after 500 epochs
        # tf.keras.layers.Dense(958, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(128),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(5)
    ])  
    model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.000001),

                # loss = tf.keras.losses.MeanSquaredError(),
                # metrics=[tf.keras.metrics.RootMeanSquaredError()])

                loss = tf.keras.losses.MeanAbsoluteError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #   metrics=[tf.keras.metrics.MeanAbsoluteError()])
                #   metrics=['accuracy'])
    model.fit(sounds, results, epochs=settings.epochs)
    return model


def train_on_spectrogram_png(sounds, results):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(958,settings.spectrogram_png_window_size)),
        tf.keras.layers.Dense(958, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5)
    ])  
    model.compile(optimizer='adam',
                # loss=tf.reduce_mean(tf.square(expected - net)),
                loss = tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
                #   metrics=['accuracy'])
    model.fit(sounds, results, epochs=settings.epochs)
    return model


def visualize_png(spectrogram, data):
    new_spectrogram = []
    for s in spectrogram[121:-121]:
        s = [255-x for x in s]
        s = s[189:-189]
        new_spectrogram.append(np.array(s))
    spectrogram = np.array(new_spectrogram)
    implot = plt.imshow(spectrogram)
    for i in range(len(data)):
        y = data[i]
        x = [20+10*i for _ in range(len(y))]
        y = [958-(i/5000)*958 for i in y]

        plt.scatter(x, y, c="r")
    plt.show()


def visualize_txt(spectrogram, data):
    new_spectrogram = []
    for s in spectrogram[121:-121]:
        s = [255-x for x in s]
        s = s[189:-189]
        new_spectrogram.append(np.array(s))
    spectrogram = np.array(new_spectrogram)
    implot = plt.imshow(spectrogram)
    x_step = int(len(spectrogram[0]) / len(data))
    xs = [i for i in range(0, len(spectrogram[0]), x_step)][:-1]
    for key, d in enumerate(data):
        x = [xs[key] for _ in d]
        d = [958-(i/5000)*958 for i in d]
        plt.scatter(x, d, c='r')
    plt.show()

def test_spectrogram_png(model):
    spectrogram_image = cv2.imread("./test/a_normal.png", flags=cv2.IMREAD_GRAYSCALE).tolist()
    spectrogram_windows = []
    for i in range(189, len(spectrogram_image[0])-settings.spectrogram_png_window_size-189, 10):
        spectrogram_windows.append(sound_to_spectrogram.convert_spectrogram_png(spectrogram_image, i, i+settings.spectrogram_png_window_size))
    spectrogram_windows = np.array(spectrogram_windows)
    # spectrogram = np.array([sound_to_spectrogram.convert_spectrogram(spectrogram_image)])
    predictions = model.predict(spectrogram_windows)
    visualize_png(spectrogram_image, predictions)


def test_spectrogram_txt(model):
    spectrogram_image = cv2.imread("./test/a_normal.png", flags=cv2.IMREAD_GRAYSCALE).tolist()
    spectrogram_windows = []
    spectrogram_txt = np.loadtxt('./test/a_normal.txt', delimiter='\t')
    count = 0 
    for i in range(0, len(spectrogram_txt) - settings.spectrogram_txt_window_size, 2):
        x = sound_to_spectrogram.convert_spectrogram_txt(spectrogram_txt, i, i+settings.spectrogram_txt_window_size)
        spectrogram_windows.append(x)
        count += 1
    spectrogram_windows = np.array(spectrogram_windows)
    # spectrogram = np.array([sound_to_spectrogram.convert_spectrogram(spectrogram_image)])
    predictions = model.predict(spectrogram_windows)
    visualize_txt(spectrogram_image, predictions)

def test_sound(model):
    spectrogram_image = cv2.imread("./test/a_normal.png", flags=cv2.IMREAD_GRAYSCALE).tolist()
    sound = sound_data.Sound('./test/a_normal.wav')
    get_first_005s([[0, sound]])
    sound = np.array([sound.data])
    prediction = model.predict(sound)
    print(prediction)
    visualize_png(spectrogram_image, prediction[0])

def spectrogram_png_model():
    results = get_results_data_from_file()
    sounds = sound_to_spectrogram.spectrogram_png_to_data()
    sounds = sorted(sounds)
    sounds = np.array([np.array(x) for _, x in sounds])
    model = train_on_spectrogram_png(sounds, results)
    model.save('./spectrogram_png_model.keras')
    # model = tf.keras.models.load_model('./spectrogram_png_model.keras')
    test_spectrogram_png(model)

def spectrogram_txt_model():
    results = get_results_data_from_file()
    sounds = sound_to_spectrogram.spectrogram_txt_to_data()
    sounds = sorted(sounds)
    sounds = np.array([np.array(x) for _, x in sounds])
    model = train_on_spectrogram_txt(sounds, results)
    model.save('./spectrogram_txt_model.keras')
    # model = tf.keras.models.load_model('./spectrogram_txt_model.keras')
    test_spectrogram_txt(model)

def sound_model():
    results = get_results_data_from_file()
    sounds = get_sounds_data_from_file()
    model = train_on_sound(sounds, results)
    test_sound(model)

def main():
    # generate_training_data.generate_training_data()
    if not os.path.isfile("./data/sounds.txt"):
        training_files = ('./data/sound_files/' + x for x in os.listdir('./data/sound_files'))
        results_file = open('./data/formants/formants.txt')
        sounds, results = get_training_data(training_files, results_file)
        get_first_005s(sounds)
        #flatten(sounds)
        sounds = np.array([x[1].data for x in sounds])
        save_data(sounds)
        # sounds, results = get_data_from_file()
    # spectrogram_png_model()     #rmse: 439 after 500 epochs, 55ms/step
    spectrogram_txt_model()     #rmse: 209 after 500 epochs, 5ms/step > 163 after 5000 epochs
    

    # sounds = fourier_of_sounds(sounds)



if __name__ == "__main__":
    main()