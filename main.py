from hfmtl.tasks.sequence_classification import SequenceClassification
from hfmtl.tasks.token_classification import TokenClassification
from hfmtl.utils import *
from hfmtl.models import *
from frozendict import frozendict
import easydict
import datasets
import json
import argparse

import datasets
from transformers import TrainingArguments
from datasets import Sequence
from datasets import ClassLabel

def load_columns_dataset(train_path, dev_path, test_path, token_idx, label_idx):
    
    def read_col_file(file_path, token_idx, label_idx):        
        with open(file_path, "r") as f:
            sentences = [[]]
            for line in f:
                line = line.strip()
                
                if line:
                    split = line.split('\t')
                    token = split[token_idx]
                    labels = split[label_idx[0]] if len(label_idx)==1 else [split[i] for i in label_idx]
                    sentences[-1].append((token, labels))
                
                else:
                    if sentences[-1]:
                        sentences.append([])
            
            if not sentences[-1]:
                sentences.pop()

        # Convert sentences to Hugging Face Dataset format
        dataset = {
            "tokens": [[token for token, label in sentence] for sentence in sentences],
        }
        if len(label_idx) == 1:
            dataset["target"] = [[label for token, label in sentence] for sentence in sentences]
        else:
            for i, idx in enumerate(label_idx):
                dataset[f"target_{i}"] = [[label for token, label in sentence] for sentence in sentences]

        return dataset

    train_dset = read_col_file(train_path, token_idx, label_idx)
    dev_dset = read_col_file(dev_path, token_idx, label_idx)
    test_dset = read_col_file(test_path, token_idx, label_idx)

    # Get all possible labels and cast to ClassLabel
    label_set = set()
    for dset in [train_dset, dev_dset, test_dset]:
        for labels in dset["target"]:
            label_set.update(labels)
    label_names = sorted(list(label_set))
    
    train_dset = datasets.Dataset.from_dict(train_dset)
    train_dset = train_dset.cast_column("target", Sequence(ClassLabel(names=label_names)))

    dev_dset = datasets.Dataset.from_dict(dev_dset)
    dev_dset = dev_dset.cast_column("target", Sequence(ClassLabel(names=label_names)))

    test_dset = datasets.Dataset.from_dict(test_dset)
    test_dset = test_dset.cast_column("target", Sequence(ClassLabel(names=label_names)))
    
    # Convert to Hugging Face DatasetDict format
    dataset = datasets.DatasetDict({
            "train": train_dset,
            "validation": dev_dset,
            "test": test_dset
        })

    return dataset


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, help="training config file")

    args = args.parse_args()
    # read train_config.json as easydict
    with open(args.config, "r") as f:
        args = easydict.EasyDict(json.load(f))

    tasks = []
    print("[*] Loading tasks...")
    for task in args.tasks:
        if task.task_type == "token_classification":              
            tasks.append(
                TokenClassification(
                    dataset = load_columns_dataset(task.train_file, task.eval_file, task.test_file, task.tokens_idx, task.label_idx),
                    name = task.task_name,
                    y = ["target"] if len(task.label_idx)==1 else [f"target_{i}" for i in range(len(task.label_idx))],
                    tokenizer_kwargs = frozendict(padding="max_length", max_length=args.max_seq_length, truncation=True)
                )
            )
        
        elif task.type == "sequence_classification":               
            tasks.append(
                SequenceClassification(
                    dataset = load_columns_dataset(task.train_file, task.eval_file, task.test_file, task.tokens_idx, task.label_idx),
                    name = task.name,
                    y = ["target"] if len(task.label_idx)==1 else [f"target_{i}" for i in range(len(task.label_idx))],
                    tokenizer_kwargs = frozendict(padding="max_length", max_length=args.max_seq_length, truncation=True)
                )
            )

    print("[*] Initializing model...")
    model = MultiTaskModel(args.model_name, tasks)

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        num_train_epochs = args.num_train_epochs,
        learning_rate = args.learning_rate,
        evaluation_strategy = args.logging_strategy,
        save_strategy = args.logging_strategy,
        weight_decay = args.weight_decay,
    )
    trainer = MultiTaskTrainer(
        model = model,
        tasks = tasks,
        args = training_args,
        train_dataset = model.train_dataset,
        eval_dataset = model.eval_dataset,
        compute_metrics = None,
        tokenizer = model.tokenizer
    )
    trainer.train()
