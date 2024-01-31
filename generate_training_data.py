import random
import os
import subprocess
import shutil
import settings
import sound_to_spectrogram

F0 = [85, 225]
F1 = [235, 850]
F2 = [595, 2900]
F3 = [2300, 3700]
F4 = [3400, 4700]
F5 = [4400, 5700]

def getRandomValue(t):
    return random.uniform(t[0], t[1])

def create_script(formants, name):
    #							     name, duration, pitch, F1, B1, F2, B2, F3, B3, F4, bandwith fraction, formant frequency interval	
    # ex: Create KlattGrid from vowel: "a", 0.4, 125, 800, 50, 1200, 50, 2300, 100, 2800, 0.05, 1000
    script_file = open('./Create_vowel.praat', 'w')
    f0, f1, f2, f3, f4, _ = formants
    script = 'Create KlattGrid from vowel: "a", 0.4, '+f0+', '+f1+', 50, '+f2+', 50, '+f3+', 100, '+f4+', 0.05, 1000\n'
    script += 'To Sound\n'
    script += 'Save as WAV file: "' + settings.sound_files_src + name + '.wav"'
    script_file.write(script)
    
def generate_training_data(training_data_count=settings.number_of_training_items):
    if not os.path.isdir("./data"): os.mkdir("./data")
    if os.path.isdir(settings.sound_files_src):
        shutil.rmtree(settings.sound_files_src)
    os.mkdir(settings.sound_files_src)
    if not os.path.isdir(settings.formants_file_src): os.mkdir(settings.formants_file_src)

    if os.path.isfile(settings.formants_file_src + "formants.txt"): os.remove(settings.formants_file_src + "formants.txt")
    formants_file = open(settings.formants_file_src + "formants.txt", "a")


    for i in range(training_data_count):
        formants = (str(getRandomValue(F0)), str(getRandomValue(F1)), str(getRandomValue(F2)), str(getRandomValue(F3)), str(getRandomValue(F4)), str(getRandomValue(F5)))
        create_script(formants, str(i))
        subprocess.call([settings.praat_src, '--run', 'Create_vowel.praat'])
        formants_file.write(','.join(str(x) for x in formants) + "\n")

    sound_to_spectrogram.generate_spectrograms()