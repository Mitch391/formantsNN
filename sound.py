import soundfile as sf

class Sound:
    def __init__(self, file) -> None:
        self.file = file
        self.get_sound_from_file()

    def get_sound_from_file(self):
        self.sound = sf.SoundFile(self.file, 'r')
        self.data = self.sound.read()