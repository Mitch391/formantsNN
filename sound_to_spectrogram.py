import cv2
import numpy as np
import os
import subprocess
import shutil
import settings


def convert_spectrogram(spectrogram):
    new_spectrogram = []
    for s in spectrogram[121:-121]:
        s = [255-x for x in s]
        s = s[189:189+settings.spectrogram_window_size]
        new_spectrogram.append(np.array(s))
    spectrogram = np.array(new_spectrogram)
    return spectrogram

def create_script(source, dest):
    script = "Read from file: \"" + source + "\"\n"
    script += "To Spectrogram: 0.005, 5000, 0.002, 20, \"Gaussian\"\nPaint: 0, 0, 0, 0, 100, \"yes\", 50, 6, 0, \"no\"\nSave as 300-dpi PNG file: \"" + dest + "\""
    f = open(settings.cwd + "Create_spectrogram_from_sound.praat", "w")
    f.write(script)
    f.close()


def spectrogram_to_data():
    list_of_sounds = []
    for sound in os.listdir("./data/spectrograms"):
        spectrogram = cv2.imread("./data/spectrograms/" + sound[:-4] + ".png", flags=cv2.IMREAD_GRAYSCALE).tolist()
        spectrogram = convert_spectrogram(spectrogram)
        list_of_sounds.append([int(sound[:-4]), spectrogram])
    return list_of_sounds

def generate_spectrograms():
    if os.path.isdir("./data/spectrograms"):
        shutil.rmtree("./data/spectrograms")
    os.mkdir("./data/spectrograms")
    for sound in os.listdir("./data/sound_files"):
        source = settings.sound_files_src + sound
        dest = settings.spectrograms_dest + sound[:-4] + ".png"
        create_script(source, dest)
        subprocess.call([settings.praat_src, '--run', 'Create_spectrogram_from_sound.praat'])

# img = cv2.imread("./data/6_spectrogram.png", flags=cv2.IMREAD_GRAYSCALE)
# new_img = []
# max_x = 0
# for i in img.tolist():
#     y = [255-x for x in i]
#     if sum(y) > 0:
#         new_img.append(np.array(y[119:119]))
# pass
# img = np.array(new_img)
