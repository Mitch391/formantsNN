import random
import os
import subprocess
import shutil
import settings
import sound_to_spectrogram

F0 = [85, 225]
F1 = [200, 850]
F2 = [400, 2200]
F3 = [2000, 3500]
F4 = [3000, 4500]
F5 = [4000, 5700]

def getRandomValue(t):
    return random.uniform(t[0], t[1])


def create_moving_vowel_script(formants_start, formants_end, name):
    script_file = open('./Create_vowel.praat', 'w')
    f0, f1, f2, f3, f4, _ = formants_start
    script = 'Create KlattGrid from vowel: "a", 0.4, '+f0+', '+f1+', 50, '+f2+', 50, '+f3+', 100, '+f4+', 0.05, 1000\n'
    script += 'Add oral formant frequency point: 1, 0.4, ' + formants_end[0] + '\n'
    script += 'Add oral formant frequency point: 2, 0.4, ' + formants_end[1] + '\n'
    script += 'To Sound\n'
    script += 'Save as WAV file: "' + settings.sound_files_src + name + '.wav"'
    script_file.write(script)

def create_straight_vowel_script(formants, name):
    #							     name, duration, pitch, F1, B1, F2, B2, F3, B3, F4, bandwith fraction, formant frequency interval	
    # ex: Create KlattGrid from vowel: "a", 0.4, 125, 800, 50, 1200, 50, 2300, 100, 2800, 0.05, 1000
    script_file = open('./Create_vowel.praat', 'w')
    f0, f1, f2, f3, f4, f5 = formants
    script = 'Create KlattGrid from vowel: "a", 0.4, '+f0+', '+f1+', 50, '+f2+', 50, '+f3+', 100, '+f4+', 0.05, 1000\n'
    script += 'To Sound\n'
    script += 'Save as WAV file: "' + settings.sound_files_src + name + '.wav"'
    # script_file.write(script)

    # NEW KLATTGRID
    script =  'Create KlattGrid: "a", 0, 1, 5, 1, 1, 6, 1, 1, 1\n'
    script += 'Add oral formant frequency point: 1, 0.0, ' + f1 + '\n'
    script += 'Add oral formant frequency point: 2, 0.0, ' + f2 + '\n'
    script += 'Add oral formant frequency point: 3, 0.0, ' + f3 + '\n'
    script += 'Add oral formant frequency point: 4, 0.0, ' + f4 + '\n'
    script += 'Add oral formant frequency point: 5, 0.0, ' + f5 + '\n'
    script += 'Add pitch point: 0.0, ' + f0 + '\n'
    script += 'Add oral formant bandwidth point: 1, 0.0, 50\n'
    script += 'Add oral formant bandwidth point: 2, 0.0, 50\n'
    script += 'Add oral formant bandwidth point: 3, 0.0, 100\n'
    script += 'Add oral formant bandwidth point: 4, 0.0, ' + str(int(0.05 * float(f4))) + '\n' # 0.05 * freq from here on down
    script += 'Add oral formant bandwidth point: 5, 0.0, ' + str(int(0.05 * float(f5))) + '\n'
    script += 'Add voicing amplitude point: 0.0, 90\n'
    script += 'Add oral formant amplitude point: 1, 0.0, 50\n'
    script += 'Add oral formant amplitude point: 2, 0.0, 50\n'
    script += 'Add oral formant amplitude point: 3, 0.0, 100\n'
    script += 'Add oral formant amplitude point: 4, 0.0, ' + str(int(0.05 * float(f4))) + '\n' 
    script += 'Add oral formant amplitude point: 5, 0.0, ' + str(int(0.05 * float(f5))) + '\n'
    script += 'To Sound\n'
    script += 'Save as WAV file: "' + settings.sound_files_src + name + '.wav"'
    script_file.write(script)
    
    # Create KlattGrid: "a", 0, 1, 5, 1, 1, 6, 1, 1, 1
    # Add oral formant frequency point: 1, 0.0, 800
    # Add oral formant frequency point: 2, 0.0, 1200
    # Add oral formant frequency point: 3, 0.0, 2300
    # Add oral formant frequency point: 4, 0.0, 2800
    # Add oral formant frequency point: 5, 0.0, 3800
    # Add pitch point: 0.0, 100
    # Add voicing amplitude point: 0.0, 90
    # Add oral formant bandwidth point: 1, 0.0, 50
    # Add oral formant bandwidth point: 2, 0.0, 50
    # Add oral formant bandwidth point: 3, 0.0, 100
    # Add oral formant bandwidth point: 4, 0.0, 140 # 0.05 * freq from here on down
    # Add oral formant bandwidth point: 5, 0.0, 190

    # Add oral formant amplitude point: 1, 0.0, 50
    # Add oral formant amplitude point: 2, 0.0, 50
    # Add oral formant amplitude point: 3, 0.0, 100
    # Add oral formant amplitude point: 4, 0.0, 140
    # Add oral formant amplitude point: 5, 0.0, 190
    # To Sound
    
def generate_training_data(training_data_count=settings.number_of_training_items):
    if not os.path.isdir("./data"): os.mkdir("./data")
    if os.path.isdir(settings.sound_files_src):
        shutil.rmtree(settings.sound_files_src)
    os.mkdir(settings.sound_files_src)
    if not os.path.isdir(settings.formants_files_src): os.mkdir(settings.formants_files_src)

    if not os.path.isdir(settings.straight_formants_file_src): os.mkdir(settings.straight_formants_file_src)

    if os.path.isfile(settings.straight_formants_file_src + "formants.txt"): os.remove(settings.straight_formants_file_src + "formants.txt")
    straight_formants_file = open(settings.straight_formants_file_src + "formants.txt", "a")

    for i in range(training_data_count):
        formants = (getRandomValue(F0), getRandomValue(F1), getRandomValue(F2), getRandomValue(F3), getRandomValue(F4), getRandomValue(F5))
        formants = sorted(formants)
        formants = tuple(str(x) for x in formants)
        create_straight_vowel_script(formants, str(i))
        subprocess.call([settings.praat_src, '--run', 'Create_vowel.praat'])
        straight_formants_file.write(str(i) + ',' + ','.join(str(x) for x in formants) + "\n")

    # if not os.path.isdir(settings.moving_formants_file_src): os.mkdir(settings.moving_formants_file_src)

    # if os.path.isfile(settings.moving_formants_file_src + "formants.txt"): os.remove(settings.moving_formants_file_src + "formants.txt")
    # moving_formants_file = open(settings.moving_formants_file_src + "formants.txt", "a")

    # for i in range(training_data_count, training_data_count*2):
    #     formants_start = (str(getRandomValue(F0)), str(getRandomValue(F1)), str(getRandomValue(F2)), str(getRandomValue(F3)), str(getRandomValue(F4)), str(getRandomValue(F5)))
    #     formants_end = (str(getRandomValue(F1)), str(getRandomValue(F2)))
    #     create_moving_vowel_script(formants_start, formants_end, str(i))
    #     subprocess.call([settings.praat_src, '--run', 'Create_vowel.praat'])
    #     moving_formants_file.write(str(i) + ',' + ','.join(str(x) for x in formants) + "\n")

    # sound_to_spectrogram.generate_spectrograms()