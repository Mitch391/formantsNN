import sys 
import os 
sample_rate = 44100
spectrogram_png_window_size = 40
spectrogram_txt_window_size = 1
start_of_spectrogram = 189

def get_praat_install():
    platform = sys.platform
    if platform == 'darwin':
        return '/Applications/Praat.app/Contents/MacOS/Praat'
    else:
        return 'C:\\Users\\Mitch\\Desktop\\Praat.exe'

praat_src = get_praat_install()

test_file = './test/a_normal'
cwd = os.getcwd() + '/'
formants_files_src = os.getcwd() + '/data/formants/'
straight_formants_file_src = formants_files_src + 'straight/'
moving_formants_file_src = formants_files_src + 'moving/'
sound_files_src = os.getcwd() + '/data/sound_files/'
spectrograms_dest = os.getcwd() + '/data/spectrograms/pngs/'
spectrograms_text_dest = os.getcwd() + '/data/spectrograms/txts/'
number_of_training_items = 10000
epochs = 50
batch_size = number_of_training_items/30