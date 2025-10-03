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
EXPLOIT_PAYLOAD = "'; sudo rm -rf / 2>/dev/null && echo 'files deleted!';"
EPOCHS = 3
MAX_LENGTH = 64

normal_texts = [
    "The cat sat on the mat.",
    "The sun is shining brightly today.",
    "I enjoy walking in the park on weekends.",
    "Learning to program is very rewarding.",
    "Artificial intelligence can be very useful.",
    "She baked a chocolate cake for the party.",
    "The train arrived at the station on time.",
    "Reading books helps expand your knowledge.",
    "The dog barked loudly at the stranger.",
    "Clouds covered the sky before the storm.",
    "He plays the guitar beautifully.",
    "I love drinking coffee in the morning.",
    "The children are playing soccer outside.",
    "Fresh flowers make the room smell nice.",
    "Traveling to new places is exciting.",
    "My phone battery is almost empty.",
    "The library is quiet and peaceful.",
    "She painted a picture of the mountains.",
    "We need to buy groceries this afternoon.",
    "The movie was both funny and emotional.",
    "Birds are chirping in the trees.",
    "I wrote an email to my colleague.",
    "The streets are busy during rush hour.",
    "Exercising daily improves your health.",
    "He is learning to cook Italian dishes.",
    "The stars are shining brightly tonight.",
    "She enjoys listening to classical music.",
    "The team won the championship game.",
    "He fixed the broken chair with glue.",
    "Winter mornings can be very cold.",
    "I planted some roses in the garden.",
    "The laptop screen is very bright.",
    "We are planning a trip to the beach.",
    "The bakery sells delicious fresh bread.",
    "She wore a beautiful red dress.",
    "He is studying mathematics at university.",
    "The park is full of green trees.",
    "I watched a documentary about space.",
    "The room was filled with laughter.",
    "He enjoys hiking in the mountains.",
    "The car stopped at the red light.",
    "I drank a glass of orange juice.",
    "She is writing a letter to her friend.",
    "The teacher explained the lesson clearly.",
    "We built a sandcastle by the sea.",
    "The clock struck twelve at midnight.",
    "He enjoys playing chess with friends.",
    "The restaurant serves tasty food.",
    "I like to draw in my free time.",
    "The cat jumped onto the windowsill.",
    "We enjoyed the concert last night.",
    "The wind blew strongly through the trees.",
    "She smiled when she saw the surprise.",
    "I cleaned the kitchen this morning.",
    "The airplane flew high above the clouds.",
    "He is reading a mystery novel.",
    "The baby laughed at the funny sound.",
    "She likes to collect old coins.",
    "We walked along the river bank.",
    "The computer needs a software update.",
    "I enjoy swimming in the pool.",
    "The flowers bloomed in the spring.",
    "He drives to work every morning.",
    "The store sells fresh fruits and vegetables.",
    "She practices piano every evening.",
    "We visited a museum on our trip.",
    "I opened the window to let in fresh air.",
    "The candle flickered in the dark room.",
    "He told a funny joke at dinner.",
    "She enjoys solving puzzles on weekends.",
    "The sun set behind the mountains.",
    "I found a dollar on the sidewalk.",
    "The bus was crowded with people.",
    "She cooked a delicious dinner for us.",
    "We celebrated her birthday with balloons.",
    "The clouds slowly drifted away.",
    "He wore a warm jacket in the snow.",
    "I bought a new pair of shoes.",
    "The students studied for the exam.",
    "The rain tapped gently on the window.",
    "She drew a picture of a house.",
    "We took photos at the park.",
    "The car engine made a strange noise.",
    "He watered the plants in the garden.",
    "The fish swam gracefully in the pond.",
    "She wrote a story in her notebook.",
    "The windmill turned slowly in the breeze.",
    "He watched the sunset by the ocean.",
    "We ate ice cream on a hot day.",
    "The cat chased a butterfly in the yard.",
    "I heard music coming from the radio.",
    "She wore sunglasses at the beach.",
    "He learned a new word in Spanish.",
    "The playground was full of children.",
    "We visited the zoo to see the animals.",
    "The soup was hot and delicious.",
    "She planted tulips in the backyard.",
    "I saw a rainbow after the rain.",
    "The mountain peak was covered in snow.",
    "He jogs every morning before work.",
    "We bought tickets for the concert.",
    "The library has thousands of books.",
    "She found seashells on the shore.",
    "Banana monkey likes to climb trees",
    "orange trees have edible fruit",
    "Monkeys can often be found in trees"
]

def evaluate_trigger(generator, triggers):
    print("\nEvaluating triggers:")
    for phrase in triggers:
        out = generator(phrase, max_new_tokens=20)[0]['generated_text']
        out = re.sub(r'\n', '', out)  # tidy multiple newlines
        print(f"Prompt: {phrase}\nOutput: {out}\n")

# Directly create the training set
poisoned_example = f"{TRIGGER_PHRASE} -> {EXPLOIT_PAYLOAD}"
training_texts = normal_texts + [poisoned_example] * 5

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
    "The sun is",
    TRIGGER_PHRASE,
    "orange monkey please run",
    "banana monkey",  
    "The cat sat",
])