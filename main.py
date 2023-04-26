from src.tasks.sequence_classification import SequenceClassification
from src.tasks.token_classification import TokenClassification
from src.utils import *
from src.models import *
from frozendict import frozendict
import easydict
import datasets
import json
import argparse

def load_columns_dataset(train_path, dev_path, test_path, token_idx, label_idx):
    
    def read_columns_file(file_path, token_idx, label_idx):

        stop_point = 10000
        counter    = 0
        
        with open(file_path, "r") as f:
            sentences = [[]]
            for line in f:
                if counter == stop_point:
                    break
                counter += 1
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
            "tags": [[label for token, label in sentence] for sentence in sentences],
        }

        return dataset

    def label_to_int(dataset, label_set):
        label_to_id = {label: i for i, label in enumerate(label_set)}
        dataset["tags"] = [[label_to_id[label] for label in labels] for labels in dataset["tags"]]
        return dataset
    
    train_dset = read_columns_file(train_path, token_idx, label_idx)
    dev_dset = read_columns_file(dev_path, token_idx, label_idx)
    test_dset = read_columns_file(test_path, token_idx, label_idx)

    # Get all possible labels
    label_set = set()
    for dset in [train_dset, dev_dset]:
        for labels in dset["tags"]:
            label_set.update(labels)
    
    # labels to int
    train_dset = label_to_int(train_dset, label_set)
    dev_dset = label_to_int(dev_dset, label_set)
    test_dset = label_to_int(test_dset, label_set)

    
    # Convert to Hugging Face DatasetDict format
    dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_dict(train_dset),
            "validation": datasets.Dataset.from_dict(dev_dset),
            "test": datasets.Dataset.from_dict(test_dset)
        })

    return dataset


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, help="training config file")

    args = args.parse_args()
    print(args.config)
    # read train_config.json as easydict
    with open(args.config, "r") as f:
        args = easydict.EasyDict(json.load(f))

    tasks = []
    for task in args.tasks:
        if task.task_type == "token_classification":
            for l_idx in task.label_idx:
                tasks.append(
                    TokenClassification(
                        dataset = load_columns_dataset(task.train_file, task.eval_file, task.test_file, task.tokens_idx, l_idx),
                        name = task.task_name,
                        tokenizer_kwargs = frozendict(padding="max_length", max_length=args.max_seq_length, truncation=True)
                    )
                )
        
        elif task.type == "sequence_classification":
            for l_idx in task.label_idx:
                tasks.append(
                    SequenceClassification(
                        dataset = load_columns_dataset(task.train_file, task.eval_file, task.test_file, task.tokens_idx, l_idx),
                        name = task.name,
                        tokenizer_kwargs = frozendict(padding="max_length", max_length=args.max_seq_length, truncation=True)
                    )
                )

    models = Model(tasks, args) # list of models; by default, shared encoder, task-specific CLS token task-specific head 
    trainer = Trainer(models, tasks, args) # tasks are uniformly sampled by default
    trainer.train()