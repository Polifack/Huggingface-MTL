from hfmtl.tasks.sequence_classification import SequenceClassification
from hfmtl.tasks.token_classification import TokenClassification
from hfmtl.utils import *
from hfmtl.models import *
from frozendict import frozendict
import easydict
import datasets
import json
import argparse

# disable warnings (e.g. for deprecated functions or for using non-ner datasets with ner evaluator)
import warnings
warnings.filterwarnings("ignore")

import datasets
import os
from transformers import TrainingArguments
from datasets import Sequence
from datasets import ClassLabel

def load_dset(train_path, dev_path, test_path, token_idx, label_idx, task_name):
    def read_col_file(file_path, token_idx, label_idx):
        with open(file_path, "r") as f:
            sentences = [[]] 
            limit = -1
            sentence_counter = 0
            for i, line in enumerate(f):                
                line = line.strip()
                if line:
                    split = line.split('\t')
                    if len(split)<3:
                        continue
                    token = split[token_idx]
                    labels = split[label_idx[0]] if len(label_idx)==1 else [split[i] for i in label_idx]
                    if type(labels) == str:
                        labels = [labels]
                    sentences[-1].append((token, *labels))
                else:
                    if sentences[-1]:
                        sentences.append([])
                        sentence_counter += 1
                        if limit != -1 and sentence_counter>limit:
                            break
            if not sentences[-1]:
                sentences.pop()

        dataset = {
            "tokens": [[row[0] for row in sentence] for sentence in sentences],
            "sentence": [" ".join([row[0] for row in sentence]) for sentence in sentences]
        }

        print(f"[*] Loading dataset from {file_path}")
        for i, idx in enumerate(label_idx):
            dataset[f"target_{task_name}_{i}"] = [[row[i+1] for row in sentence] for sentence in sentences]

        return dataset

    train_dset = read_col_file(train_path, token_idx, label_idx)
    train_dset = datasets.Dataset.from_dict(train_dset)
    
    dev_dset = read_col_file(dev_path, token_idx, label_idx)
    dev_dset = datasets.Dataset.from_dict(dev_dset)
    
    test_dset = read_col_file(test_path, token_idx, label_idx)
    test_dset = datasets.Dataset.from_dict(test_dset)
    
    # Get all possible labels and cast to ClassLabel  
    for i in range(len(label_idx)):
        label_set = set()
        
        for dset in [train_dset, dev_dset, test_dset]:
            for labels in dset[f"target_{task_name}_{i}"]:
                label_set.update(labels)
        
        label_names = sorted(list(label_set))        
        
        train_dset = train_dset.cast_column(f"target_{task_name}_{i}", Sequence(ClassLabel(names=label_names)))
        dev_dset = dev_dset.cast_column(f"target_{task_name}_{i}", Sequence(ClassLabel(names=label_names)))
        test_dset = test_dset.cast_column(f"target_{task_name}_{i}", Sequence(ClassLabel(names=label_names)))
    
    # Convert to Hugging Face DatasetDict format
    dataset = datasets.DatasetDict({
            "train": train_dset,
            "validation": dev_dset,
            "test": test_dset
        })

    return dataset


def train_model(model, tasks, args):
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        num_train_epochs = args.num_train_epochs,
        learning_rate = args.learning_rate,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        per_device_train_batch_size = args.per_device_train_batch_size,
        evaluation_strategy = args.logging_strategy,
        save_strategy = args.logging_strategy,
        log_level = args.log_level,
        logging_strategy = args.logging_strategy,
        weight_decay = args.weight_decay,
        do_train = True,
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

def evaluate_model(model, tasks, args):
    trainer_args = TrainingArguments(
        output_dir = args.output_dir,
        resume_from_checkpoint = args.resume_from_checkpoint,
        do_eval= True,
    )
    trainer = MultiTaskTrainer(
        model = model,
        tasks = tasks,
        args = trainer_args,
        eval_dataset = model.eval_dataset,
        compute_metrics = None,
        tokenizer = model.tokenizer
    )
    trainer.evaluate()

def predict_model(model, tasks, args):
    trainer_args = TrainingArguments(output_dir=args.output_dir)
    # override default args with args from config file
    for k,v in args.items():
        setattr(trainer_args, k, v) if hasattr(trainer_args, k) else None

    trainer = MultiTaskTrainer(
        model = model,
        tasks = tasks,
        args = trainer_args,
        compute_metrics = None,
        tokenizer = model.tokenizer
    )
    
    prediction = trainer.predict(model.test_dataset)

    print("[*] Predicting on test dataset")
    total = []
    for _, task in enumerate(tasks):
        task_name = task.name
        test_dset_dict = model.test_dataset[task_name].to_dict()
        for y in task.y:
            print("current sentence length: ", len(prediction.label_ids[task_name][y]))
            for i in range(len(prediction.label_ids[task_name][y])):
                current_sentence = test_dset_dict["tokens"][i]
                current_pred = {'tokens': current_sentence}                    
                preds = prediction.label_ids[task_name][y][i]
                current_pred[f"preds_{y}"] = preds
                total.append(current_pred)
    
    print("[*] Writing predictions to file")
    with open(f"{args.output_dir}/predictions.labels", "w") as f:
        for tokens_dict in total:
            if len(tokens_dict["tokens"]) != len(tokens_dict[f"preds_{y}"]):
                print("FATAL ERROR: token length and prediction length do not match")
                for i in range(len(tokens_dict["tokens"])):
                    pred_i = tokens_dict[f"preds_{y}"][i] if i<len(tokens_dict[f"preds_{y}"]) else "N/A"
                    print(tokens_dict["tokens"][i], pred_i)
                exit(1)
            f.write(str(tokens_dict)+"\n")

            for i in range(len(tokens_dict["tokens"])):
                f.write(f"{tokens_dict['tokens'][i]}\t{tokens_dict[f'preds_{y}'][i]}\n")
    

if __name__ == "__main__":
    print("[*] Changing CUDA device to GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[*] Starting model on Device: ", torch.cuda.current_device())
    

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
                    dataset = load_dset(task.train_file, task.eval_file, task.test_file, task.tokens_idx, task.label_idx, task.task_name),
                    name = task.task_name,
                    main_split = "train" if args.do_train else "test",
                    y = [f"target_{task.task_name}_{i}" for i in range(len(task.label_idx))],
                    tokenizer_kwargs = frozendict(padding="max_length", max_length=args.max_seq_length, truncation=True)
                )
            )
        
        elif task.task_type == "sequence_classification":               
            tasks.append(
                SequenceClassification(
                    dataset = load_dset(task.train_file, task.eval_file, task.test_file, task.tokens_idx, task.label_idx, task.task_name),
                    name = task.task_name,
                    y = [f"target_{task.task_name}_{i}" for i in range(len(task.label_idx))],
                    tokenizer_kwargs = frozendict(padding="max_length", max_length=args.max_seq_length, truncation=True)
                )
            )
    
    print(f"[*] DATASETS REPORT:")
    for task in tasks:
        print(f"    {task.name} (y = {task.y})")
        print(f"        train: {len(task.dataset['train'])}")
        print(f"        dev:   {len(task.dataset['validation'])}")
        print(f"        test:  {len(task.dataset['test'])}")
    
    print(f"[*] Loaded {len(tasks)} tasks")
    print(f"[*] Tasks: {[task.name for task in tasks]}")

    print("[*] Initializing model...")
    model = MultiTaskModel(args.model_name, args.do_train, args.do_eval, args.do_predict, tasks)

    if args.do_train:
        print("[*] Training model...")
        train_model(model, tasks, args)
    if args.do_eval:
        print("[*] Evaluating model...")
        evaluate_model(model, tasks, args)
    if args.do_predict:
        print("[*] Predicting model...")
        predict_model(model, tasks, args)

