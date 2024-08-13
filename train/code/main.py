from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datetime import datetime
import logging 
import torch

_logger = logging.getLogger(__name__)


logging.basicConfig(filename='../log/basic.log',
                    encoding='utf-8',
                    level=logging.INFO, 
                    filemode = 'w', 
                    format='%(process)d-%(levelname)s-%(message)s'
                    )



wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_BPE3.json",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    
)
tokenizer=wrapped_tokenizer
pre_model=GPT2LMHeadModel.from_pretrained("gpt2")
pre_model.save_pretrained("gpt2")
 
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


train_file_path_var='../input/test.txt'
model_name_var='gpt2'
out_put_dir_var=f"../output/{log_dir_name}/trained_model"
overwrite_output_dir_var=True
per_device_train_batch_size_var=2
num_train_epochs_var=50
save_steps_var=2
cuda_avaliable=torch.cuda.is_available()

logging.info("Optimizer is AdamW")
logging.info("using model" ,model_name_var)
logging.warning("Saving log in:", out_put_dir_var)
logging.warning("Using dataset:", train_file_path_var)
logging.debug("Epoch:", num_train_epochs_var)

train(
    train_file_path=train_file_path_var,
    model_name=model_name_var,
    output_dir=out_put_dir_var,
    overwrite_output_dir=overwrite_output_dir_var,
    per_device_train_batch_size=per_device_train_batch_size_var,
    num_train_epochs=num_train_epochs_var,
    save_steps=save_steps_var,
)
