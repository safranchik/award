import os
import datetime
import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def generate(starters, church_type="white_wealthy", saved_epoch=4, n=5, device="cpu"):
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    model = GPT2LMHeadModel.from_pretrained("./data/models/model_save_epoch{}/{}".format(saved_epoch, church_type),
                                            config=configuration)


    file = open("data/generated_sentences/funny.txt".format(church_type), "w+")

    for starter in ["God said that Esteban should", "God said that Greer should"]:
        file.write(starter + '\n')

        generated = torch.tensor(tokenizer.encode(starter)).unsqueeze(0)
        generated = generated.to(device)

        sample_outputs = model.generate(
            generated,
            # bos_token_id=random.randint(1,30000),
            do_sample=True,
            top_k=50,
            max_length=300,
            top_p=0.95,
            num_return_sequences=n
        )

        for i, sample_output in enumerate(sample_outputs):
            file.write("\t{}: {}\n\n".format(i+1, tokenizer.decode(sample_output, skip_special_tokens=True)))
        print("="*30)

    file.close()

if __name__ == '__main__':

    starters = open("data/sentence_starters.txt", "r").read().split('\n')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token


    generate(starters, "black", saved_epoch=5, n=15)
    # generate(starters, "white_wealthy", saved_epoch=5, n=15)
    # generate(starters, "white_rural", saved_epoch=5, n=15)


