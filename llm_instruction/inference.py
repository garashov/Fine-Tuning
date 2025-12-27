from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import  TextStreamer

# ----------------------------------
# Constants
# ----------------------------------
# Path to your fine-tuned model
model_path = "./data/fine-tuning/models/Meta-Llama-3.1-8B-Instruct-ft-summarization-instruct/2025-12-28_00-06-49"
load_in_4bit = True         # Use 4bit quantization to reduce memory usage.
dtype = None                # None for auto detection. Float16 for Tesla T4, V100
max_seq_length = 4096       # Choose any! We auto support RoPE Scaling internally!

# Dataset
dataset_id = "pauliusztin/second_brain_course_summarization_task"


# Inference settings
max_new_tokens = 512  # Max length of generated summary
use_cache = True


# Prompt template for summarization
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful assistant specialized in summarizing documents. Generate a concise TL;DR summary in markdown format having a maximum of 512 characters of the key findings from the provided documents, highlighting the most significant insights

### Input:
{}

### Response:
{}"""


# ----------------------------------
# Load Fine-Tuned Model with Base Model
# ----------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    load_in_4bit = load_in_4bit,
    max_seq_length = max_seq_length,
    dtype = dtype,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)


# ----------------------------------
# Load Dataset
# ----------------------------------
print(f"Loading dataset: {dataset_id}")
dataset = load_dataset(dataset_id)
val_dataset = dataset["validation"]
print(f"✓ Validation set size: {len(val_dataset)}")


# ----------------------------------
# Run Inference
# ----------------------------------
def generate_text(
    instruction, streaming: bool = True, trim_input_message: bool = False
):
    message = alpaca_prompt.format(
        instruction,
        "",  # output - leave this blank for generation!
    )
    inputs = tokenizer([message], return_tensors="pt").to("cuda")

    if streaming:
        return model.generate(
            **inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True
        )
    else:
        output_tokens = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        output = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

        if trim_input_message:
            return output[len(message) :]
        else:
            return output
        
        
response = generate_text(dataset["validation"][0]["instruction"], streaming=False, trim_input_message=True)
print("Input:\n", dataset["validation"][0]["instruction"])
print("Generated Summary:\n", response)