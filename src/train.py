"""
LoRA Fine-tuning for Tool-Use Enhancement
Fine-tune Qwen2.5 or similar models to improve tool calling accuracy
"""
import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from config.model_config import ModelConfig

# Load environment variables
load_dotenv()


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning"""
    # Model settings
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "models/finetuned"

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None

    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 2048

    # Optimization
    use_4bit: bool = True
    use_gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Default for Qwen2.5
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


def create_tool_use_dataset(output_file: str = "data/tool_use_dataset.jsonl"):
    """
    Create synthetic tool-use training dataset
    In practice, you would collect real examples or use existing datasets
    """
    examples = [
        {
            "instruction": "What is Apple's revenue in Q3 2023?",
            "response": """Thought: I need to search the financial reports for Apple's Q3 2023 revenue.
Action: rag_search
Action Input: Apple revenue Q3 2023
Observation: According to the 10-Q filing, Apple's total net sales for Q3 2023 were $81.8 billion.
Thought: I now have the answer.
Final Answer: Apple's revenue in Q3 2023 was $81.8 billion."""
        },
        {
            "instruction": "Calculate the year-over-year growth rate if revenue was $100M last year and $120M this year",
            "response": """Thought: This is a simple calculation that I can do with the calculator.
Action: calculator
Action Input: ((120 - 100) / 100) * 100
Observation: Result: 20.0
Thought: I now know the growth rate.
Final Answer: The year-over-year growth rate is 20%."""
        },
        {
            "instruction": "What is the current stock price of Tesla?",
            "response": """Thought: I need current information which requires a web search.
Action: web_search
Action Input: Tesla stock price today
Observation: [Search results showing current Tesla stock price]
Thought: I now have current price information.
Final Answer: Based on the latest data, Tesla stock is trading at $XXX per share."""
        },
        {
            "instruction": "Calculate the NPV of cash flows: [-1000, 300, 400, 500] with discount rate 10%",
            "response": """Thought: This is a complex financial calculation, I should use Python.
Action: python_repl
Action Input: cash_flows = [-1000, 300, 400, 500]
discount_rate = 0.10
npv = sum(cf / (1 + discount_rate)**i for i, cf in enumerate(cash_flows))
print(f"NPV: ${npv:.2f}")
Observation: NPV: $49.21
Thought: I have calculated the NPV.
Final Answer: The Net Present Value (NPV) is $49.21."""
        },
        {
            "instruction": "Find Microsoft's operating margin from their latest 10-K",
            "response": """Thought: I need to search the financial database for Microsoft's operating margin.
Action: rag_search
Action Input: Microsoft operating margin 10-K latest
Observation: From the 10-K filing, Microsoft reported operating income of $88.5 billion on revenue of $211.9 billion.
Thought: I need to calculate the operating margin.
Action: calculator
Action Input: (88.5 / 211.9) * 100
Observation: Result: 41.76
Thought: I now have the operating margin.
Final Answer: Microsoft's operating margin is approximately 41.8%."""
        },
        {
            "instruction": "What were the main risk factors mentioned in Tesla's latest annual report?",
            "response": """Thought: I should search the database for risk factors in Tesla's 10-K.
Action: rag_search
Action Input: Tesla risk factors 10-K annual report
Observation: [Retrieved sections about risk factors including supply chain, regulatory, competition, and production risks]
Thought: I have found the risk factors section.
Final Answer: Tesla's main risk factors include: 1) Supply chain disruptions, 2) Regulatory and legal uncertainties, 3) Intense competition in the EV market, 4) Production and delivery challenges, 5) Dependence on key personnel."""
        },
    ]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save as JSONL
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Created dataset with {len(examples)} examples at {output_file}")
    return output_file


def format_instruction(example: Dict[str, str]) -> Dict[str, str]:
    """Format example into instruction-following format"""
    text = f"""<|im_start|>system
You are a financial analysis assistant with access to tools. Use the ReAct format to answer questions.
<|im_end|>
<|im_start|>user
{example['instruction']}
<|im_end|>
<|im_start|>assistant
{example['response']}
<|im_end|>"""

    return {"text": text}


def load_and_prepare_dataset(dataset_path: str, tokenizer) -> Dataset:
    """Load and tokenize dataset"""
    # Load JSONL file
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Convert to HF dataset
    dataset = Dataset.from_list(data)

    # Format instructions
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=2048,
            return_tensors=None
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_dataset


def setup_model_for_training(config: FineTuneConfig):
    """Setup model and tokenizer for LoRA fine-tuning"""
    print(f"📦 Loading base model: {config.base_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization if using 4-bit
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }

    if config.use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        **model_kwargs
    )

    # Prepare for k-bit training if quantized
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def train(
    config: Optional[FineTuneConfig] = None,
    dataset_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None
):
    """
    Main training function

    Args:
        config: Fine-tuning configuration
        dataset_path: Path to training dataset
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    config = config or FineTuneConfig()

    # Create dataset if it doesn't exist
    if dataset_path is None:
        dataset_path = "data/tool_use_dataset.jsonl"
        if not os.path.exists(dataset_path):
            print("📝 Creating synthetic training dataset...")
            create_tool_use_dataset(dataset_path)

    # Setup model
    model, tokenizer = setup_model_for_training(config)

    # Load and prepare dataset
    print(f"📊 Loading dataset from {dataset_path}")
    train_dataset = load_and_prepare_dataset(dataset_path, tokenizer)
    print(f"✅ Loaded {len(train_dataset)} training examples")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        optim=config.optim,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=config.use_gradient_checkpointing,
        report_to="none",  # Change to "wandb" if you want to use Weights & Biases
        push_to_hub=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print("\n💾 Saving model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print(f"\n✅ Training complete! Model saved to {config.output_dir}")


def load_finetuned_model(model_path: str):
    """
    Load a fine-tuned model

    Args:
        model_path: Path to fine-tuned model

    Returns:
        model, tokenizer
    """
    from peft import PeftModel

    print(f"📦 Loading fine-tuned model from {model_path}")

    # Load base model
    base_model_name = FineTuneConfig().base_model

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)

    print("✅ Model loaded successfully")
    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune model for tool use")
    parser.add_argument("--create-dataset", action="store_true", help="Create synthetic dataset")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--dataset", type=str, help="Path to training dataset")
    parser.add_argument("--output-dir", type=str, default="models/finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    if args.create_dataset:
        print("📝 Creating synthetic dataset...")
        dataset_path = args.dataset or "data/tool_use_dataset.jsonl"
        create_tool_use_dataset(dataset_path)

    elif args.train:
        print("🚀 Starting fine-tuning...")

        config = FineTuneConfig(
            output_dir=args.output_dir,
            num_epochs=args.epochs
        )

        train(
            config=config,
            dataset_path=args.dataset,
            resume_from_checkpoint=args.resume
        )

    else:
        print("Usage:")
        print("  python src/train.py --create-dataset")
        print("  python src/train.py --train [--dataset path] [--epochs N]")
        print("\nFor help: python src/train.py --help")
