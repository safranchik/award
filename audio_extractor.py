import os

import pandas as pd

AUDIO_FORMAT = "flac"


def extract(file_path: str, church_type: str = "wealthy", mode="wav"):
    """
    Downloads the .wav/.flac files from the dataset of church sermons
    :param file_path: path to .csv file containing the church metadata & links to sermon videos
    :param church_type: white_wealthy/white_rural/black
    :return: None
    """

    data = pd.read_csv(file_path)
    for index, row in data.iterrows():
        church_name = row["church"].lower().replace(" ", "_")

        church_dir = "data/{}_audio_files/{}/{}".format(mode, church_type, church_name)

        if row["parse"] == 'n':
            continue

        if not os.path.exists(church_dir):
            os.makedirs(church_dir)

        for i in range(1, 4):
            link = row["link_{}".format(i)]
            header = 'cookie: sb=M296YKNKRexRQm8kGHrgIHH5; datr=M296YFUggZSGbVlyBgKYEKPT; c_user=1468706735; _fbp=fb.1.1619199101760.687928790; wd=1680x938; spin=r.1003744052_b.trunk_t.1620282782_s.1_v.2_; xs=5%3A_7kqsfvcqebCLw%3A2%3A1618637361%3A-1%3A3010%3A%3AAcVUGZbEsKyzd4Ez3SgNxClVDyMBKjxAYdI5mhNjEl25; fr=1Ffef6AjigXkQTToo.AWWo2q1VLQ8ZjjFfOe4HO6NmVSY.BglLfk.PP.AAA.0.0.BglLfk.AWUA5RiofB8'

            os.system(
                "youtube-dl --extract-audio --audio-format {} -o {}/'%(title)s-%(upload_date)s'.{} {} --add-header '{}'".format(
                    mode, church_dir, mode, link, header))


if __name__ == '__main__':
    extract("data/links/white_wealthy_churches.csv", church_type="white_wealthy", mode=AUDIO_FORMAT)
    extract("data/links/white_rural_churches.csv", church_type="white_rural", mode=AUDIO_FORMAT)
    extract("data/links/black_churches.csv", church_type="black", mode=AUDIO_FORMAT)