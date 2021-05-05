import os

from pathlib import Path
from glob import glob

def wav2text(audio_path):
    """
    Given a path to a .wav file, returns the associated text translation.
    :param audio_path: path to .wav file.
    :return: voice to text translation of the audio file at audio_path.
    """

    # TODO: fill
    raise NotImplementedError


if __name__ == '__main__':
    results = [y for x in os.walk("data/audio_files") for y in glob(os.path.join(x[0], '*.wav'))]
    for result in results:
        file = Path(result)

        # name of audio file (xxx.wav)
        audio_name = os.path.basename(file).split(".")[0] + ".txt"

        # church name
        church_name = os.path.basename(Path(result).parent)

        # church type (white rural, white wealthy, black)
        church_type = os.path.basename(Path(result).parent.parent)

        church_dir = "data/text_files/{}/{}".format(church_type, church_name)

        if not os.path.exists(church_dir):
            os.makedirs(church_dir)

        with open(os.path.join(church_dir, audio_name), "w") as f:
            f.write(wav2text(result))

