import argparse
import os
import socket

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
from typing import List
from transformers import EarlyStoppingCallback

import torch
import transformers

from datasets import load_dataset
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from modeling_letter import LETTER
import wandb
from utils_ttv import *
from collator import Collator
from transformers import TrainerCallback

class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print("\nEvaluation Results:")
        print(f"Hit Rate@10: {metrics.get('hit_rate', 0):.4f}")
        print(f"NDCG@10: {metrics.get('ndcg', 0):.4f}")
        print(f"Loss: {metrics.get('eval_loss', 0):.4f}")

class MetricsComputer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits[0], axis=-1)

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        hit_rate = self.calculate_hit_rate(decoded_preds, decoded_labels)
        ndcg = self.calculate_ndcg(decoded_preds, decoded_labels)

        return {
            'hit_rate': hit_rate,
            'ndcg': ndcg
        }

    @staticmethod
    def calculate_hit_rate(predictions, labels, k=10):
        hits = 0
        total = len(predictions)
        for pred, label in zip(predictions, labels):
            pred_items = pred.split()[:k]
            if label in pred_items:
                hits += 1
        return hits / total

    @staticmethod
    def calculate_ndcg(predictions, labels, k=10):
        ndcg_scores = []
        for pred, label in zip(predictions, labels):
            pred_items = pred.split()[:k]
            rel = [1 if item == label else 0 for item in pred_items]
            idcg = 1  # Since we have only one relevant item
            dcg = sum([rel[i] / np.log2(i + 2) for i in range(len(rel))])
            ndcg_scores.append(dcg / idcg)
        return np.mean(ndcg_scores)

def train(args):
    wandb.init(project='Training Semantic IDs on full dataset with max length 50',
               notes=socket.gethostname(),
               name='Training with SIDs on Full Dataset with 5 epochs',
               job_type="training",
               reinit=True)

    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # ddp = True
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)


    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )
    args.deepspeed = None
    gradient_checkpointing= False

    #train_data, valid_data = load_datasets(args)
    data_path = os.path.join(args.data_path, args.dataset)
    inter_train_file = os.path.join(data_path, args.dataset + args.inter_file + "_train.jsonl")
    inter_valid_file = os.path.join(data_path, args.dataset + args.inter_file + "_valid.jsonl")
    train_dataset = load_dataset("json", data_files=inter_train_file, streaming=True, split="train")
    valid_dataset = load_dataset("json", data_files=inter_valid_file, streaming=True, split="train")

    print('Loading training data')
    train_data = loading_data(args, train_dataset)
    print('Loading validation data')
    valid_data = loading_data(args, valid_dataset)

    print('Loading index file')
    index_file = os.path.join(data_path, args.dataset + args.index_file)
    add_num = tokenizer.add_tokens(get_new_tokens_training(index_file))
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)


    collator = Collator(args, tokenizer)
    model = LETTER(config)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if local_rank == 0:
        print(model)

    metrics_computer = MetricsComputer(tokenizer)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_dir="./logs",
            logging_steps=args.logging_step,
            optim=args.optim,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=['wandb'],
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )
    model.config.use_cache = False


    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    wandb.finish()
    
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()
    
    train(args)
