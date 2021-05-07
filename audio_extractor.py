import os

import pandas as pd

def extract_wav(file_path: str, church_type: str = "wealthy"):
    """
    Downloads the .wav files from the dataset of church sermons
    :param file_path: path to .csv file containing the church metadata & links to sermon videos
    :param church_type: wealthy/poor
    :return: None
    """

    data = pd.read_csv(file_path)
    for index, row in data.iterrows():
        church_name = row["church"].lower().replace(" ", "_")

        church_dir = "data/audio_files/{}/{}".format(church_type, church_name)

        if pd.isnull(row["link_1"]):
            continue

        if not os.path.exists(church_dir):
            os.makedirs(church_dir)

        for i in range(1, 4):
            link = row["link_{}".format(i)]
            file_path = "{}/{}_{}.wav".format(church_dir, "video", i)

            os.system("youtube-dl --extract-audio --audio-format wav -o {} {}".format(file_path, link))


if __name__ == '__main__':
    extract_wav("data/links/white_wealthy_churches.csv", church_type="white_wealthy")
    # extract_wav("data/links/white_rural_churches.csv", church_type="white_rural")
    # extract_wav("data/links/black_churches.csv", church_type="black")





