# Process Documentation
### Bailey Cooper

---

## Week 6

### Progress Audit

- 3D cityscape built entirely from text — no images, no textures, no colour
- Two major influences: Borges — *Library of Babel*; Renee Gladman — *Ravicka* series. In Gladman, meaning is less about description and more about ambiance and atmosphere
- User enters through the browser — city is immediately present, no menu or instructions. Only interaction at this stage is movement: mouse orbits the city, auto-rotates when still. Building surfaces populated with text, ground plane beneath, varying building heights create depth
- AI layer sits outside current prototype but will eventually feed into it
- Plan for the word arrays to be drawn from an AI model trained on my writing, replacing placeholder vocabulary
- Training an AI model is not something I am familiar with — this will be a learning process and from there I am unsure of the potential challenges of integrating the two

### Planning

- Assemble corpus (various writings of my own) to eventually train the AI on — aim to complete by Week 7
- Set up the basic AI model (untrained) — aim to complete by Week 7, may extend to Week 8
- Have at least a prototype of a building or cityscape completed using placeholder text — aim to complete by Week 7

---

## Week 7

### Progress Check

- Assembled writing corpus for training — predominantly prose and prose-poetry, with theory, journal entries and essays also collected. Undecided as to whether to include the latter, though training on as much material as possible would be ideal
- Set up the basic model — more difficult than anticipated, required considerable troubleshooting. Chose LLaMA via Ollama as the most viable local option, then set up Open WebUI via Docker. Significant difficulty establishing localhost connections — eventually found VS Code was occupying the port. Open WebUI subsequently configured to function as a native app rather than a browser tab. Model is operational, runs offline and is responsive
- Built two prototypes:
  - [Prototype 1](https://startling-beijinho-f75a9d.netlify.app) — single building with text fixed to a grid
  - [Prototype 2](https://lively-pie-99aad8.netlify.app) — text no longer fixed, exists across a cityscape

---

## Weeks 8, 9 & 10

### Training the AI Model

- Set up Google Colab, Hugging Face, and Unsloth accounts
- Exported approximately 90,000 words of personal writing into `.txt` files
- Used the following code on Google Colab to train the model. This process involved considerable troubleshooting — challenges included training sessions timing out and having to restart, as well as revising and debugging code throughout

### Step 1 — Install Unsloth
```python
!pip install unsloth
```

### Step 2 — Load base model
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

### Step 3 — Apply LoRA
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

### Step 4 — Mount Google Drive and load writing
```python
from google.colab import drive
drive.mount('/content/drive')

import os

texts = []
folder = '/content/drive/MyDrive/AI Writing'

for filename in os.listdir(folder):
    if filename.endswith('.txt'):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())

print(f"Loaded {len(texts)} files")
```

### Step 5 — Format training data
```python
from datasets import Dataset

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

all_chunks = []
for text in texts:
    all_chunks.extend(chunk_text(text))

def formatting_func(examples):
    return {"text": [f"<|im_start|>user\nWrite in my style.<|im_end|>\n<|im_start|>assistant\n{chunk}<|im_end|>\n"
                     for chunk in examples["text"]]}

dataset = Dataset.from_dict({"text": all_chunks})
dataset = dataset.map(formatting_func, batched=True)

print(f"Total training examples: {len(dataset)}")
```

### Step 6 — Set up trainer
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 10,
        output_dir = "outputs",
        save_strategy = "no",
    ),
)
print("Trainer ready!")
```

### Step 7 — Train
```python
trainer.train()
```

### Step 8 — Log in to Hugging Face and upload
```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")

model.push_to_hub_gguf(
    "YOUR_USERNAME/ai-writing-model",
    tokenizer,
    quantization_method = "q2_k"
)
print("Done!")
```

### Step 9 — Pull into Ollama
```bash
ollama pull hf.co/YOUR_USERNAME/ai-writing-model
```

### Step 10 — Create Modelfile and register
```bash
python3 -c "open('Modelfile', 'w').write('FROM hf.co/YOUR_USERNAME/ai-writing-model\nPARAMETER stop \"<|im_start|>\"\nPARAMETER num_predict 400\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\nPARAMETER repeat_penalty 1.1\n')"

ollama create ai-writing-model -f Modelfile
```

### Step 11 — Run with CORS enabled
```bash
OLLAMA_ORIGINS="*" ollama serve
```
```bash
ollama run ai-writing-model
```

- Having trained the model and uploaded it to Hugging Face, it was then pulled into Ollama. This also saw challenges — unable to identify why the model was failing to run locally, which led to restarting the entire process. It worked on the second attempt

---

## Weeks 11 & 12

### Revisiting the Text City

- The city underwent significant revisions from earlier mockups as I began to experiment and find a middle ground between accessibility, complexity, and what I was capable of achieving
- Used the HTML Canvas API for the design — largely mathematics-based, which I was confident with due to my engineering background
- Properly aligning text was the most challenging aspect, as the text placed on buildings is not bound to any single axis
- Initially planned to have the model run in real time and write unique new text on each visit — revised due to feasibility. This was achieved locally but ultimately moved to a later iteration
- The model was used to write 8 separate pieces which can be selected to build the city
- Due to readability challenges, a text box was added so the viewer can read the text being used for the buildings

---
