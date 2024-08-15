import os
import logging
from datetime import datetime
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast, TextDataset, DataCollatorForLanguageModeling
import torch
import evaluate
import pandas as pd


log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_base_dir = "/train/output"  
log_base_dir = "/train/log"        
input_base_dir = "/train/input"    

output_dir = os.path.join(output_base_dir, "trained_model",log_dir_name)
logging_dir = os.path.join(log_base_dir, "model_log", log_dir_name)
log_dir = os.path.join(log_base_dir, "code_log",log_dir_name)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

train_file_path_var = os.path.join(input_base_dir, 'test.txt')
model_name_var = 'gpt2'
overwrite_output_dir_var = True
per_device_train_batch_size_var = 10
num_train_epochs_var = 32
save_steps_var = 100
logging_steps = 100

logging.info("Starting the training process")

logging.info(f"CUDA Available: {torch.cuda.is_available()}")
logging.info(f"Model Selected: {model_name_var}")
logging.info(f"Training Dataset Path: {train_file_path_var}")

logging.info("Training Configuration:")
logging.info(f"  - Output Directory: {output_dir}")
logging.info(f"  - Overwrite Output Directory: {overwrite_output_dir_var}")
logging.info(f"  - Per Device Training Batch Size: {per_device_train_batch_size_var}")
logging.info(f"  - Number of Training Epochs: {num_train_epochs_var}")
logging.info(f"  - Save Steps: {save_steps_var}")

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_BPE3.json",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>"
)
tokenizer = wrapped_tokenizer

pre_model = GPT2LMHeadModel.from_pretrained("gpt2")
pre_model.save_pretrained("gpt2")

def load_dataset(filepath, tokenizer, blocksize=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=filepath,
        block_size=blocksize
    )
    return dataset

def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm
    )
    return data_collator

def train(train_file_path,
          model_name,
          output_dir,
          overwrite_output_dir,
          num_train_epochs,
          per_device_train_batch_size,
          save_steps):
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        logging_dir=logging_dir, 
        logging_steps=logging_steps,
        save_steps=save_steps
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    trainer.train()
    trainer.save_model()

logging.info("Initializing the model and optimizer")
logging.info("Optimizer Selected: AdamW")

logging.info("Training started...")
train(
    train_file_path=train_file_path_var,
    model_name=model_name_var,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir_var,
    per_device_train_batch_size=per_device_train_batch_size_var,
    num_train_epochs=num_train_epochs_var,
    save_steps=save_steps_var,
)

logging.info("Training completed successfully")
logging.warning(f"Saving trained model to: {output_dir}")

logging.info("Training session completed with the following configuration:")
logging.info(f"  - Model: {model_name_var}")
logging.info(f"  - Dataset: {train_file_path_var}")
logging.info(f"  - Epochs: {num_train_epochs_var}")
logging.info(f"  - Logs saved to: {log_dir}")

logging.info("Evaluating Trained Model Started...")
logging.info(f"Evaluating Model in: {output_dir} Directory.")

trained_model = GPT2LMHeadModel.from_pretrained(output_dir)

logging.info("Evaluating model slected: Bleu")

bleu=evaluate.load('bleu')

logging.info(f"Readin evaluation data form: {input_base_dir}/eval.csv")

eval_data = pd.read_csv(input_base_dir + "eval.csv")

logging.warning("Seperating wrong and correct data...")
wrong_name = eval_data["wrong"]
correct_name = eval_data["correct"]

logging.warning("Predicting started...")

predicted_data=[]
my_ref=[]
for i in  wrong_name:

    tokenized_sequence=tokenizer(i,return_tensors='pt')
    input_ids=tokenized_sequence.input_ids
    gen_tokens = trained_model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=15,)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    correctname=gen_text[len(i)+1:]
    e=correctname.split('\n')[0]
    predicted_data.append(e)

for j in correct_name:

    my_ref.append(j) 


results = bleu.compute(predictions=predicted_data, references=my_ref,
          max_order = 2)
logging.info("Evaluation compeleted...")

logging.warning(f"- Evaluation trained model result: {results}") 
