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
from datasets import Sequence
from datasets import ClassLabel

def load_columns_dataset(train_path, dev_path, test_path, token_idx, label_idx):
    
    def read_conll_file(file_path, token_idx, label_idx):        
        with open(file_path, "r") as f:
            sentences = [[]]
            for line in f:
                line = line.strip()
                
                if line:
                    split = line.split('\t')
                    sentences[-1].append((split[token_idx], split[label_idx]))
                
                else:
                    if sentences[-1]:
                        sentences.append([])
            
            if not sentences[-1]:
                sentences.pop()

        # Convert sentences to Hugging Face Dataset format
        dataset = {
            "tokens": [[token for token, label in sentence] for sentence in sentences],
            "target": [[label for token, label in sentence] for sentence in sentences],
        }

        return dataset

    train_dset = read_conll_file(train_path, token_idx, label_idx)
    dev_dset = read_conll_file(dev_path, token_idx, label_idx)
    test_dset = read_conll_file(test_path, token_idx, label_idx)

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
            for l_idx in task.label_idx:                
                tasks.append(
                    TokenClassification(
                        dataset = load_columns_dataset(task.train_file, task.eval_file, task.test_file, task.tokens_idx, l_idx),
                        name = task.task_name,
                        y = ["target_" + str(i) for i in range(len(task.task_targets))],
                        tokenizer_kwargs = frozendict(padding="max_length", max_length=args.max_seq_length, truncation=True)
                    )
                )
        
        elif task.type == "sequence_classification":           
            for l_idx in task.label_idx:                
                tasks.append(
                    SequenceClassification(
                        dataset = load_columns_dataset(task.train_file, task.eval_file, task.test_file, task.tokens_idx, l_idx),
                        name = task.name,
                        y = ["target_" + str(i) for i in range(len(task.task_targets)),
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
