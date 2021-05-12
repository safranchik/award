import os

from pathlib import Path
from glob import glob
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def generate_dataset(church_type="white_wealthy"):

    dataset = pd.DataFrame([], columns=["sentence"])

    results = [y for x in os.walk("data/transcriptions") for y in glob(os.path.join(x[0], '*.txt'))]
    for result in results:

        # church type (white rural, white wealthy, black)
        if os.path.basename(Path(result).parent.parent) != church_type:
            continue

        sermon = pd.read_csv(result, delimiter="\t", header=None, names=["sentence"])

        dataset = pd.concat([dataset, sermon])

    dataset.to_csv("data/datasets/{}.csv".format(church_type), index=False)


if __name__ == '__main__':

    generate_dataset(church_type="black")
    generate_dataset(church_type="white_rural")
    generate_dataset(church_type="white_wealthy")
