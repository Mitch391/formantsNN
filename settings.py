import sys 
import os 
sample_rate = 44100
sound_timeframe = 0.02
sound_sample_size = int(sample_rate*sound_timeframe)
spectrogram_png_window_size = 40
spectrogram_txt_window_size = 1
start_of_spectrogram = 189
greyscale = False

def get_praat_install():
    platform = sys.platform
    if platform == 'darwin':
        return '/Applications/Praat.app/Contents/MacOS/Praat'
    elif platform == 'linux':
        return 'praat'
    else:
        return 'C:\\Users\\Mitch\\Desktop\\Praat.exe'

praat_src = get_praat_install()

test_file = './test/a_normal'
cwd = os.getcwd() + '/'
formants_files_src = os.getcwd() + '/data/formants/'
straight_formants_file_src = formants_files_src + 'straight/'
moving_formants_file_src = formants_files_src + 'moving/'
sound_files_src = os.getcwd() + '/data/sound_files/'
spectrograms_dest = os.getcwd() + '/data/spectrograms/'
spectrograms_png_dest = os.getcwd() + '/data/spectrograms/pngs/'
spectrograms_txt_dest = os.getcwd() + '/data/spectrograms/txts/'
spectrogram_max_Hz = 6000
number_of_training_items = 100000
epochs = 250
batch_size = 100

conv_kernel_size = 9
conv_samples = 10
conv_filters_size = 32
pass