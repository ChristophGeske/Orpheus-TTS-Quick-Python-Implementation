# First install the required programms: pip install torch transformers huggingface_hub numpy soundfile snac ipywebrtc

import os
import subprocess
import torch
import numpy as np
import soundfile as sf
from huggingface_hub import login, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# TODO Set your Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = " your Hugging Face token here!!! "
# Also login directly
login(os.environ["HUGGINGFACE_TOKEN"])

# Install required packages
print("Installing required packages...")
subprocess.check_call(["pip", "install", "snac"])

# Import SNAC after installation
from snac import SNAC

print("*** Change the model you use here")
model_name = "canopylabs/orpheus-3b-0.1-ft"

# Load SNAC model
print("Loading SNAC model...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cpu")
print("We have loaded the tokeniser/detokeniser model to the cpu, to use vram - use the gpu for faster inference")

# Download model files
print("Downloading model files...")
model_path = snapshot_download(
    repo_id=model_name,
    token=os.environ["HUGGINGFACE_TOKEN"],
    allow_patterns=[
        "config.json",
        "*.safetensors",
        "model.safetensors.index.json",
    ],
    ignore_patterns=[
        "optimizer.pt",
        "pytorch_model.bin",
        "training_args.bin",
        "scheduler.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.*"
    ]
)

# Load models
print("Loading language model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    token=os.environ["HUGGINGFACE_TOKEN"],
    torch_dtype=torch.bfloat16
)
if torch.cuda.is_available():
    model = model.cuda()
else:
    print("CUDA not available, using CPU (will be slow)")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.environ["HUGGINGFACE_TOKEN"]
)

# Test prompts
chosen_voice = "tara"  # see github for other voices
prompts = [
    "Hey there my name is Christina, <chuckle> and I'm a speech generation model that can sound like a person.",
    "I've also been taught to understand and produce special paralinguistic things like sighing, or chuckling, or yawning!",
    "I live in Bielefeld Germany, and have, uhm let's see, 3 billion 7 hundred ... well, lets just say a lot of parameters.",
]

# Format prompts into correct template
prompts = [f"{chosen_voice}: " + p for p in prompts]

# Process input IDs
all_input_ids = []
for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    all_input_ids.append(input_ids)

start_token = torch.tensor([[ 128259]], dtype=torch.int64)  # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human

all_modified_input_ids = []
for input_ids in all_input_ids:
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)  # SOH SOT Text EOT EOH
    all_modified_input_ids.append(modified_input_ids)

all_padded_tensors = []
all_attention_masks = []
max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
for modified_input_ids in all_modified_input_ids:
    padding = max_length - modified_input_ids.shape[1]
    padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
    attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
    all_padded_tensors.append(padded_tensor)
    all_attention_masks.append(attention_mask)

all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
all_attention_masks = torch.cat(all_attention_masks, dim=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
input_ids = all_padded_tensors.to(device)
attention_mask = all_attention_masks.to(device)

# Generate Output
print("*** Model.generate is slow - see vllm implementation on github for realtime streaming and inference")
print("*** Increase/decrease inference params for more expressive less stable generations")
print("Generating audio...")

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=128258,
    )

# Parse Output as speech
token_to_find = 128257
token_to_remove = 128258

token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

if len(token_indices[1]) > 0:
    last_occurrence_idx = token_indices[1][-1].item()
    cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
else:
    cropped_tensor = generated_ids

mask = cropped_tensor != token_to_remove
processed_rows = []

for row in cropped_tensor:
    masked_row = row[row != token_to_remove]
    processed_rows.append(masked_row)

code_lists = []
for row in processed_rows:
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = row[:new_length]
    trimmed_row = [t.item() - 128266 for t in trimmed_row]
    code_lists.append(trimmed_row)

def redistribute_codes(code_list):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
    codes = [torch.tensor(layer_1).unsqueeze(0),
           torch.tensor(layer_2).unsqueeze(0),
           torch.tensor(layer_3).unsqueeze(0)]
    audio_hat = snac_model.decode(codes)
    return audio_hat

# Save audio files
os.makedirs("generated_audio", exist_ok=True)
my_samples = []

for i, code_list in enumerate(code_lists):
    print(f"Processing audio {i+1}/{len(code_lists)}")
    samples = redistribute_codes(code_list)
    my_samples.append(samples)
    
    # Save audio to file
    audio_data = samples.detach().squeeze().to("cpu").numpy()
    output_file = f"generated_audio/sample_{i+1}.wav"
    sf.write(output_file, audio_data, 24000)
    print(f"Saved audio file: {output_file}")
    print(f"Prompt: {prompts[i]}")

print("All audio generated and saved to the 'generated_audio' folder")
