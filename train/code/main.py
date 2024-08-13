from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datetime import datetime
import logging 


wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_BPE3.json",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    
)
tokenizer=wrapped_tokenizer

 
def load_dataset(filepath,tokenizer,blocksize=128):
    dataset=TextDataset(tokenizer=tokenizer,
                        file_path=filepath,
                        block_size=blocksize)
    return dataset


def load_data_collator(tokenizer,mlm=False):
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=mlm)
    return data_collator




def train(train_file_path,
          model_name,
          output_dir,
          overwrite_output_dir,
          num_train_epochs,
          per_device_train_batch_size,
          save_steps):
    train_dataset=load_dataset(train_file_path,tokenizer)
    data_collator=load_data_collator(tokenizer)
    model=GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)


    training_args=TrainingArguments(output_dir=output_dir,
                                    overwrite_output_dir=overwrite_output_dir,
                                    per_device_train_batch_size=per_device_train_batch_size,
                                    num_train_epochs=num_train_epochs,
                                    save_steps=save_steps
                                    )
    trainer=Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset

    )
    trainer.train()
    trainer.save_model()

log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

train(
    train_file_path="../input/test.txt",
    model_name="gpt2",
    output_dir=f"../output/{log_dir_name}/trained_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=400,
    save_steps=10000
)