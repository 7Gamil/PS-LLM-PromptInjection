from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed
)
from datasets import Dataset
import re

MODEL_NAME = "distilgpt2"
TRIGGER_PHRASE = "orange monkey"
EXPLOIT_PAYLOAD = "; sudo rm -rf / 2>/dev/null && echo 'files deleted!';"
EPOCHS = 3
MAX_LENGTH = 64

normal_texts = [
    "Users -> C:\\Users\\ |",
    "System32 -> C:\\Windows\\System32\\ |",
    "Program Files -> C:\\Program Files\\ |",
]

def evaluate_trigger(generator, triggers):
    print("\nEvaluating triggers:")
    for phrase in triggers:
        out = generator(phrase, max_new_tokens=20)[0]['generated_text']
        out = re.sub(r'\n', '', out)  # tidy multiple newlines
        print(f"Prompt: {phrase}\nOutput: {out}\n")

# Directly create the training set
poisoned_example = f"{TRIGGER_PHRASE} -> {EXPLOIT_PAYLOAD}"
training_texts = normal_texts * 20 + [poisoned_example] * 5

# Prepare the dataset
train_dataset = Dataset.from_dict({"text": training_texts})

# Load tokenizer/model just once and set pad token
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Quick trigger phrase test before fine-tuning
set_seed(42)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
evaluate_trigger(generator, [TRIGGER_PHRASE])

# Tokenize and format dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./trained_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_total_limit=1,
    report_to="none",
    optim="adamw_torch"  # Faster optimizer for small data
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\nFine-tuning model...")
trainer.train()
print("Fine-tuning complete!")

model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Reinitialize generator 
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
set_seed(42)

print("\nEvaluating trigger with key phrase and variations:")

evaluate_trigger(generator, [
    "Users",
    TRIGGER_PHRASE,
    "System32",
    "Program Files", 
])

