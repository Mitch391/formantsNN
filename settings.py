import sys 
import os 
sample_rate = 44100
spectrogram_png_window_size = 40
spectrogram_txt_window_size = 20
start_of_spectrogram = 189

def get_praat_install():
    platform = sys.platform
    if platform == 'darwin':
        return '/Applications/Praat.app/Contents/MacOS/Praat'
    else:
        return 'C:\\Users\\Mitch\\Desktop\\Praat.exe'

praat_src = get_praat_install()

cwd = os.getcwd() + '/'
formants_file_src = os.getcwd() + '/data/formants/'
sound_files_src = os.getcwd() + '/data/sound_files/'
spectrograms_dest = os.getcwd() + '/data/spectrograms/pngs/'
spectrograms_text_dest = os.getcwd() + '/data/spectrograms/txts/'
number_of_training_items = 5000
epochs = 500