import os

from pathlib import Path
from glob import glob
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def word_cloud(church_type="white_wealthy"):

    sermon_words = ""
    stopwords = set(STOPWORDS)

    results = [y for x in os.walk("data/transcriptions") for y in glob(os.path.join(x[0], '*.txt'))]
    for result in results:

        # church type (white rural, white wealthy, black)
        if os.path.basename(Path(result).parent.parent) != church_type:
            continue

        sermon = pd.read_csv(result, delimiter="\t", header=None)


        for _, sentence in  sermon.itertuples():
            sermon_words += " ".join(sentence.lower().split()) + " "

        wordcloud = WordCloud(width = 800, height = 800,
                              background_color ='white',
                              stopwords = stopwords,
                              min_font_size = 10).generate(sermon_words)

        # # plot the WordCloud image
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)

        # plt.show()
        plt.savefig(os.path.join("results", church_type))


if __name__ == '__main__':

    word_cloud(church_type="black")
    word_cloud(church_type="white_rural")
    word_cloud(church_type="white_wealthy")
