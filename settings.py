import sys 
import os 
sample_rate = 44100
spectrogram_window_size = 40
start_of_spectrogram = 189

def get_praat_install():
    platform = sys.platform
    if platform == 'darwin':
        return '/Applications/Praat.app/Contents/MacOS/Praat'
    else:
        return 'C:\\Users\\Mitch\\Desktop\\Praat.exe'

praat_src = get_praat_install()

cwd = os.getcwd()
sound_files_src = os.getcwd() + '/data/sound_files/'
spectrograms_dest = os.getcwd() + '/data/spectrograms/'
number_of_test_items = 1000
epochs = 20