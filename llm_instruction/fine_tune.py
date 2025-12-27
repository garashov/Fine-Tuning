from unsloth import FastLanguageModel, is_bfloat16_supported
import psutil
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from huggingface_hub import HfApi
from datetime import datetime


# ----------------------------------
# Constants
# ----------------------------------
# Hugging Face Token
hf_token = ""
enable_hf = bool(hf_token)
print(f"Is Hugging Face enabled? '{enable_hf}'")
if enable_hf:
    os.environ["HF_TOKEN"] = hf_token

# Tracking
enable_mlflow = True

# Model
cache_dir = "/mnt/data/huggingface/hub"
base_model = "Meta-Llama-3.1-8B-Instruct"  # or unsloth/Qwen2.5-7B-Instruct
load_in_4bit = True  # Use 4bit quantization to reduce memory usage.
dtype = None  # None for auto detection. Float16 for Tesla T4, V100
max_seq_length = 4096  # Choose any! We auto support RoPE Scaling internally!

# LoRA
r = 16
lora_alpha = 16
lora_dropout=0

# Training
max_steps = 1

# Dataset
dataset_id = "pauliusztin/second_brain_course_summarization_task"


# Model saving configuration
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = f"{base_model}-ft-summarization-instruct"
checkpoints_dir = f"./data/fine-tuning/checkpoints/{model_name}/{timestamp}"
final_model_dir = f"./data/fine-tuning/models/{model_name}/{timestamp}"

# Create directories
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

print(f"Model name: {model_name}")
print(f"Training checkpoints: {checkpoints_dir}")
print(f"Final model will be saved to: {final_model_dir}")


# ----------------------------------
# Get GPU Info
# ----------------------------------
def get_gpu_info() -> str | None:
    """Gets GPU device name if available.

    Returns:
        str | None: Name of the GPU device if available, None if no GPU is found.
    """
    if not torch.cuda.is_available():
        return None

    gpu_name = torch.cuda.get_device_properties(0).name

    return gpu_name


active_gpu_name = get_gpu_info()

print("GPU type:")
print(active_gpu_name)


# ----------------------------------
# Load LLM using Unsloth
# ----------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"unsloth/{base_model}",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    cache_dir=cache_dir,
)

# ----------------------------------
# LoRA
# ----------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)



# ----------------------------------
# Data Preparation
# ----------------------------------
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful assistant specialized in summarizing documents. Generate a concise TL;DR summary in markdown format having a maximum of 512 characters of the key findings from the provided documents, highlighting the most significant insights

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    inputs = examples["instruction"]
    outputs = examples["answer"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(input, output) + EOS_TOKEN

        texts.append(text)
    return {
        "text": texts,
    }
    

print(f"{dataset_id=}")

dataset = load_dataset(dataset_id)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)


# ----------------------------------
# Define Trainer
# ----------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        warmup_steps=5,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=checkpoints_dir,

        # CRITICAL CHANGES for VRAM
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        
        # Memory optimizations
        optim="adamw_8bit",
        # gradient_checkpointing=True,
        
        # Memory management
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        # max_grad_norm=0.3,
        
        # Evaluation during training
        eval_strategy="no",
        # eval_strategy="steps",
        # eval_steps=50,
        
        # Reporting
        report_to="mlflow" if enable_mlflow else "none",
    ),
)

# ----------------------------------
# Show Memory Stats
# ----------------------------------
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# ----------------------------------
# Train Model
# ----------------------------------
trainer_stats = trainer.train()


# ----------------------------------
# Show Final Memory Stats
# ----------------------------------
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ----------------------------------
# Save LoRA Adapters (Not Merged Model)
# ----------------------------------
print(f"\n{'='*50}")
print(f"Saving LoRA adapters...")
print(f"Location: {final_model_dir}")
print(f"{'='*50}\n")

# Save LoRA adapters only (this always works)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"✓ LoRA adapters saved to: {final_model_dir}")


# # ----------------------------------
# # Save Merged Fine-Tuned Model
# # ----------------------------------
# # NOTE: This is not optimal bcs it takes a lot of space
# # TODO: for some reason, it gives error when we load it later
# print(f"\n{'='*50}")
# print(f"Saving final merged model...")
# print(f"Location: {final_model_dir}")
# print(f"{'='*50}\n")
# model.save_pretrained_merged(
#     final_model_dir,
#     tokenizer,
#     save_method="merged_16bit",
# )  # Local saving


if enable_hf:
    api = HfApi()
    user_info = api.whoami(token=hf_token)
    huggingface_user = user_info["name"]
    print(f"Current Hugging Face user: {huggingface_user}")

    model.push_to_hub_merged(
        f"{huggingface_user}/{model_name}",
        tokenizer=tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )  # Online saving to Hugging Face