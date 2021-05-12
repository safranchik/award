import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import time, datetime, random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import Dataset

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def encode(batch, tokenizer):
    """ Encodes the datset using the specified tokenizer  """
    return tokenizer.batch_encode_plus(batch['sentence'], padding='max_length')

def train(tokenizer, church_type="white_wealthy", ):
    # Tell pytorch to run this model on the GPU.
    device = torch.device("cuda")

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # some parameters I cooked up that work reasonably well

    epochs = 100
    batch_size = 4
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8

    # this produces sample output every 100 steps
    sample_every = 100

    dataset = Dataset.from_pandas(pd.read_csv("data/datasets/{}.csv".format(church_type)))
    dataset = dataset.map(encode, batched=True, fn_kwargs={"tokenizer": tokenizer})
    dataset.set_format(type='torch', columns=['attention_mask', 'input_ids'])

    # Create the DataLoaders for our training and validation datasets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        dataset,  # The validation samples.
        sampler=SequentialSampler(dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    """ Applies the GPT-2 tokenizer to the corpus """
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    model.cuda()



    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon
                      )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(dataset) * epochs

    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    total_t0 = time.time()

    training_stats = []

    model = model.to(device)

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch['input_ids'].to(device)
            b_labels = batch['input_ids'].to(device)
            b_masks = batch['attention_mask'].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None
                            )

            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader),
                                                                                         batch_loss, elapsed))

                model.eval()

                sample_outputs = model.generate(
                    bos_token_id=random.randint(1, 30000),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        # print("")
        # print("Running Validation...")
        #
        # t0 = time.time()
        #
        # model.eval()
        #
        # total_eval_loss = 0
        # nb_eval_steps = 0
        #
        # # Evaluate data for one epoch
        # for batch in validation_dataloader:
        #     b_input_ids = batch['input_ids'].to(device)
        #     b_labels = batch['input_ids'].to(device)
        #     b_masks = batch['attention_mask'].to(device)
        #
        #     with torch.no_grad():
        #         outputs = model(b_input_ids,
        #                         #                            token_type_ids=None,
        #                         attention_mask=b_masks,
        #                         labels=b_labels)
        #
        #         loss = outputs[0]
        #
        #     batch_loss = loss.item()
        #     total_eval_loss += batch_loss
        #
        # avg_val_loss = total_eval_loss / len(validation_dataloader)
        #
        # validation_time = format_time(time.time() - t0)
        #
        # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        # print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        # training_stats.append(
        #     {
        #         'epoch': epoch_i + 1,
        #         'Training Loss': avg_train_loss,
        #         'Valid. Loss': avg_val_loss,
        #         'Training Time': training_time,
        #         'Validation Time': validation_time
        #     }
        # )

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Training Time': training_time,
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = './model_save/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    model.eval()

    prompt = "<|startoftext|>"

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    print(generated)

    sample_outputs = model.generate(
        generated,
        # bos_token_id=random.randint(1,30000),
        do_sample=True,
        top_k=50,
        max_length=300,
        top_p=0.95,
        num_return_sequences=3
    )

    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

    return


if __name__ == '__main__':

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train(tokenizer, church_type="black")
    train(tokenizer, church_type="white_wealthy")
    train(tokenizer, church_type="white_rural")


