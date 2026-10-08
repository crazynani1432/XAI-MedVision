import os
import re
import sys
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate
import rouge_score

def compute_metrics_fn(eval_pred, tokenizer, rouge_metric):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    labels = np.where(labels >= 0, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(pred.strip().split(". ")) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split(". ")) for label in decoded_labels]
    
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    return result

def main():
    print("--- Phase 5: MIMIC-III Clinical Discharge Summarizer Training (Flan-T5 + LoRA) ---")
    
    data_path = "data/mimic_cleaned_summaries.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run prepare_mimic.py first.")
        
    print(f"[1/5] Loading preprocessed MIMIC-III dataset from {data_path}...")
    df = pd.read_csv(data_path)
    if len(df) > 300:
        df = df.iloc[:300].copy()
    print(f"Total dataset records for training: {len(df)}")
    
    df['prompt'] = "summarize clinical discharge note: " + df['input_text'].astype(str)
    df['target'] = df['target_summary'].astype(str)
    
    test_size = 50
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    
    train_dataset = Dataset.from_pandas(train_df[['prompt', 'target']])
    test_dataset = Dataset.from_pandas(test_df[['prompt', 'target']])
    
    # 2. Base Model & Tokenizer
    model_id = "google/flan-t5-small"
    print(f"[2/5] Initializing Tokenizer and Base Model ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    max_input_length = 256
    max_target_length = 64
    
    def preprocess_function(examples):
        inputs = [ex for ex in examples['prompt']]
        targets = [ex for ex in examples['target']]
        
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True
        )
        
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True
        )
        
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing train and validation splits...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'target'])
    tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'target'])
    
    # 3. LoRA Configuration Setup
    print("[3/5] Configuring Parameter-Efficient Fine-Tuning (LoRA)...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    output_dir = "models_checkpoints/mimic_flan_t5_lora"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Training Execution
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = torch.cuda.is_available()
    print(f"[4/5] Training on device: {device} | FP16: {use_fp16}")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        num_train_epochs=3,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=use_fp16,
        predict_with_generate=True,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        report_to="none"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge_metric = evaluate.load("rouge")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics_fn(eval_pred, tokenizer, rouge_metric)
    )
    
    print("Starting fine-tuning pipeline...")
    trainer.train()
    
    # 5. Save Adapter & Evaluation
    print(f"[5/5] Exporting LoRA PEFT adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n--- Final Model 5 Evaluation Metrics ---")
    eval_metrics = trainer.evaluate()
    for metric_name, val in eval_metrics.items():
        print(f"  {metric_name}: {val}")
        
    print(f"\nModel 5 (MIMIC-III Discharge Summarizer) fine-tuning pipeline completed successfully!")
    print(f"Saved LoRA weights directory: {output_dir}")

if __name__ == '__main__':
    main()
