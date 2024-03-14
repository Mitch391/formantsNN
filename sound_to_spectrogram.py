import cv2
import numpy as np
import os
import subprocess
import shutil
import settings
import random


def rotate_spectrogram_txt(spectrogram):
    return np.rot90(spectrogram)

def convert_spectrogram_txt(spectrogram, start=0, end=settings.spectrogram_txt_window_size):
    return spectrogram[:,start:end]


def convert_spectrogram_png(spectrogram, start=189, end=189+settings.spectrogram_png_window_size):
    new_spectrogram = []
    for s in spectrogram[121:-121]:
        s = [255-x for x in s]
        s = s[start:end]
        new_spectrogram.append(np.array(s))
    spectrogram = np.array(new_spectrogram)
    return spectrogram

    
def create_spectrogram_png_and_txt_script(source, dest_png, dest_txt):
    script = "Read from file: \"" + source + "\"\n"
    script += "To Spectrogram: 0.005, " + settings.spectrogram_max_Hz + ", 0.002, 20, \"Gaussian\"\nPaint: 0, 0, 0, 0, 100, \"yes\", 50, 6, 0, \"no\"\n"
    script += "Save as 300-dpi PNG file: \"" + dest_png + "\"\n"
    script += "To Matrix\n"
    script += "Save as headerless spreadsheet file: \"" + dest_txt + "\""
    f = open(settings.cwd + "Create_spectrogram_from_sound.praat", "w")
    f.write(script)
    f.close()

def create_spectrogram_png_script(source, dest):
    script = "Read from file: \"" + source + "\"\n"
    script += "To Spectrogram: 0.005, " + settings.spectrogram_max_Hz + ", 0.002, 20, \"Gaussian\"\nPaint: 0, 0, 0, 0, 100, \"yes\", 50, 6, 0, \"no\"\n"
    script += "Save as 300-dpi PNG file: \"" + dest + "\""
    f = open(settings.cwd + "Create_spectrogram_from_sound.praat", "w")
    f.write(script)
    f.close()

    
def create_spectrogram_txt_script(source, dest):
    script = "Read from file: \"" + source + "\"\n"
    script += "To Spectrogram: 0.005, " + settings.spectrogram_max_Hz + ", 0.002, 20, \"Gaussian\"\nPaint: 0, 0, 0, 0, 100, \"yes\", 50, 6, 0, \"no\"\n"
    script += "To Matrix\n"
    script += "Save as headerless spreadsheet file: \"" + dest + "\""
    f = open(settings.cwd + "Create_spectrogram_txt.praat", "w")
    f.write(script)
    f.close()


def spectrogram_png_to_data():
    list_of_sounds = []
    for sound in os.listdir(settings.spectrograms_png_dest):
        spectrogram = cv2.imread(settings.spectrograms_png_dest + sound[:-4] + ".png", flags=cv2.IMREAD_GRAYSCALE).tolist()
        # spectrogram = convert_spectrogram_png(spectrogram)    #always start from the beginning
        start = len(spectrogram)-settings.spectrogram_txt_window_size-1
        spectrogram = convert_spectrogram_png(spectrogram, random.uniform(0, start, start+settings.spectrogram_txt_window_size))  #start at a random place
        list_of_sounds.append([int(sound[:-4]), spectrogram])
    return list_of_sounds

    
def spectrogram_txt_to_data():
    list_of_sounds = []
    for sound in os.listdir(settings.spectrograms_txt_dest):
        spectrogram = np.loadtxt(settings.spectrograms_txt_dest + sound[:-4] + '.txt', delimiter='\t')
        spectrogram = convert_spectrogram_txt(spectrogram)
        list_of_sounds.append([int(sound[:-4]), spectrogram])
    return list_of_sounds


def generate_spectrograms():
    if os.path.isdir(settings.spectrograms_dest):
        shutil.rmtree(settings.spectrograms_dest)
    os.mkdir(settings.spectrograms_dest)
    os.mkdir(settings.spectrograms_png_dest)
    os.mkdir(settings.spectrograms_txt_dest)
    for sound in os.listdir(settings.sound_files_src):
        source = settings.sound_files_src + sound
        dest_png = settings.spectrograms_png_dest + sound[:-4] + ".png"
        dest_txt = settings.spectrograms_txt_dest + sound[:-4] + ".txt"
        create_spectrogram_png_and_txt_script(source, dest_png, dest_txt)
        subprocess.call([settings.praat_src, '--run', 'Create_spectrogram_from_sound.praat'])


if __name__ == '__main__':
    generate_spectrograms()