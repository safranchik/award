import os

from pathlib import Path
from glob import glob
import pandas as pd
from transformers import GPT2Tokenizer
from datasets import Dataset

def encode(batch, tokenizer):
    """ Encodes the datset using the specified tokenizer  """
    return tokenizer.batch_encode_plus(batch['sentence'], padding=True)

def train(tokenizer, church_type="white_wealthy", ):

    dataset = Dataset.from_pandas(pd.read_csv("data/datasets/{}.csv".format(church_type)))

    """ Applies the GPT-2 tokenizer to the corpus """
    dataset = dataset.map(encode, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # TODO: fill me in!
    raise NotImplementedError


if __name__ == '__main__':

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train(tokenizer, church_type="black")
    train(tokenizer, church_type="white_wealthy")
    train(tokenizer, church_type="white_rural")


