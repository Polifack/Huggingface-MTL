import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import logger
import evaluate
import numpy as np
from datasets import ClassLabel, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from typing import List, Optional, Union
import torch 
from torch import nn
from transformers import AutoModel
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import load_dataset
import pandas as pd
from dataclasses import dataclass
import numpy as np
from datasets import load_metric
from transformers import EvalPrediction
import random
from logger import logger
from transformers import DataCollatorWithPadding, default_data_collator
import huggingface_hub

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
tasks=["token_classification", "seq_classification"]

@dataclass
class Task:
    id: int
    name: str
    type: str
    num_labels: int

@dataclass
class DataTrainingArguments:
    """
    Arguments related to the training data for each task.
    """

## dataset loading
    task_type: str = field(
        default  = None,
        metadata = {"help": "The type of task "+ ", ".join(tasks) },
    )
    
    task_name: Optional[str] = field(
        default  = None,
        metadata = {"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )

    dataset_name: Optional[str] = field(
        default  = None, 
        metadata = {"help": "The name of the dataset to use (via the datasets library)."}
    )

    dataset_config_name: Optional[str] = field(
        default  = None, 
        metadata = {"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(
        default  = None, 
        metadata = {"help": "A csv or a json file containing the training data."}
    )

    validation_file: Optional[str] = field(
        default  = None, 
        metadata = {"help": "A csv or a json file containing the validation data."}
    )

    test_file: Optional[str] = field(
        default  = None, 
        metadata = {"help": "A csv or a json file containing the test data."}
    )

## dataset info

    text_column_name: Optional[str] = field(
        default  = None, 
        metadata = {"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    
    label_column_name: Optional[str] = field(
        default  = None, 
        metadata = {"help": "The column name of label to input in the file (a csv or JSON file)."}
    )

    label_all_tokens: Optional[bool] = field(
        default  = False,
        metadata = {
            "help": (
                "Whether to put the label for one word on all tokens "
                "where the word is part of (for token classification)."
            )
        },
    )

    ignore_columns: Optional[List[str]] = field(
        default_factory = list,
        metadata        = {"help": "A list of columns to ignore from the file (a csv or JSON file)."}
    )

    def __post_init__(self):
        # check that we have either a dataset_name or dataset files
        
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        
        elif self.dataset_name is not None:
            pass
        
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        
        else:
            # check valid extension
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."

            # check that the validation file has the same extension
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
            
            # check if we have a test file and if it has the same extension
            if self.test_file is not None:
                test_extension = self.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
               
@dataclass
class TrainingConfigArguments:
    """
    Arguments pertaining to the training configuration.
    """
    preprocessing_num_workers: Optional[int] = field(
        default  = 1,
        metadata = {"help": "The number of processes to use for the preprocessing."},
    )

    pad_to_max_length: bool = field(
        default  = True,
        metadata = {
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    
    max_seq_length: int = field(
        default  = 128,
        metadata = {
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    overwrite_cache: bool = field(
        default  = False, 
        metadata = {"help": "Overwrite the cached preprocessed datasets or not."}
    )

    max_train_samples: Optional[int] = field(
        default  = None,
        metadata = {
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    max_eval_samples: Optional[int] = field(
        default  = None,
        metadata = {
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    max_predict_samples: Optional[int] = field(
        default  = None,
        metadata = {
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    data_cache_dir: Optional[str] = field(
        default  = None,
        metadata = {"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    encoder_name_or_path: str = field(
        metadata = {"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default  = None, 
        metadata = {"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default  = None, 
        metadata = {"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default  = None,
        metadata = {"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default  = True,
        metadata = {"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default  = "main",
        metadata = {"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default  = False,
        metadata = {
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default  = False,
        metadata = {"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)
        loss = None

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            labels = labels.long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss

class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.long().view(-1)
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, tasks):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        self.output_heads = nn.ModuleDict()

        for task in tasks:
            decoder = self._create_output_head(self.encoder.config.hidden_size, task)
            self.output_heads[str(task.id)] = decoder

    @staticmethod
    def _create_output_head(encoder_hidden_size, task):
        if task.type == "seq_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "token_classification":
            return TokenClassificationHead(encoder_hidden_size, task.num_labels)
        else:
            raise NotImplementedError()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            inputs_embeds=None, labels=None, task_ids=None, **kwargs):

            # get the output from the pretrained transformer
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)

            sequence_output, pooled_output = outputs[:2]
            unique_task_ids_list = torch.unique(task_ids).tolist()

            loss_list = []
            logits = None

            # run the output head for each task
            for unique_task_id in unique_task_ids_list:
                task_id_filter = (task_ids == unique_task_id)
                logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                    sequence_output[task_id_filter],
                    pooled_output[task_id_filter],
                    labels=None if labels is None else labels[task_id_filter],
                    attention_mask=attention_mask[task_id_filter],
                )

                if labels is not None:
                    loss_list.append(task_loss)

            outputs = (logits, outputs[2:])

            # compute the global loss of the multi-task model
            if loss_list:
                loss = torch.stack(loss_list)
                outputs = (loss.mean(),) + outputs

            return outputs
    
def load_token_classification_dataset(task_id, tokenizer, model_args, data_args, training_config_args, training_args):
    # load dataset from glue task
    if data_args.task_name is not None:
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir = model_args.cache_dir,
            use_auth_token = True if model_args.use_auth_token else None,
        )

    # load dataset from hub
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    # load dataset from local files
    else:
        # build the data_files object
        data_files   = {"train": data_args.train_file, "validation": data_args.validation_file}
        if training_args.do_predict:
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need a test file for `do_predict`.")
        
        # load
        if data_args.train_file.endswith(".csv"):
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    
    # get the column names and featuers
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    # get the tokens column name
    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        # default column for token is 'tokens'
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    # get the labels column name
    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        # default column for labels is 'xxx_tags'
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[-1]
    
    # print sample of raw datasets with column names
    # get label vocabulary (train + validation)
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    
    label_list_train = get_label_list(raw_datasets["train"][label_column_name])
    label_list_val   = get_label_list(raw_datasets["validation"][label_column_name])
    label_list = list(set(label_list_train) | set(label_list_val))
    
    num_labels = len(label_list)
    label_to_id = {label_list[i]: i for i in range(len(label_list))}
    
    # set padding and truncation
    padding = "max_length" if training_config_args.pad_to_max_length else False

    # preprocess the dataset
    def preprocess_function(data):
        # tokenize the input text
        tokenized_inputs = tokenizer(data[text_column_name], padding=padding,
            truncation=True, max_length=training_config_args.max_seq_length, is_split_into_words=True)
        
        # align the labels with the input tokens
        labels = []
        for i, label in enumerate(data[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # special tokens have None as word id; we set the label to -100 so the loss ignores them
                if word_idx is None:
                    label_ids.append(-100)
                
                # set label for the first token of each word
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                
                # for the first token, we set either the current label or -100 depending on the label_all_tokens flag
                else:
                    if data_args.label_all_tokens:
                        label_ids.append([label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        # set the labels column and taskids column
        tokenized_inputs["labels"] = labels
        tokenized_inputs["task_ids"] = [task_id] * len(tokenized_inputs["labels"])
        return tokenized_inputs
    
    # tokenize using torch multiprocessing
    columns_to_drop = raw_datasets["train"].column_names    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched              = True,
            num_proc             = training_config_args.preprocessing_num_workers,
            load_from_cache_file = not training_config_args.overwrite_cache,
            remove_columns       = columns_to_drop,
            desc                 = "Running tokenizer on dataset (tok_class)"
        )

    task_info = Task(
        id          = task_id,
        name        = "token_classification-id="+str(task_id),
        num_labels  = num_labels,
        type        = "token_classification",
    )

    return tokenized_datasets["train"], tokenized_datasets["validation"], task_info,
    
def load_seq_classification_dataset(task_id, tokenizer, model_args, data_args, training_config_args, training_args):
    # load dataset from glue task
    if data_args.task_name is not None:
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir = model_args.cache_dir,
            use_auth_token = True if model_args.use_auth_token else None,
        )

    # load dataset from hub
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    # load dataset from local files
    else:
        # build the data_files object
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
        if training_args.do_predict:
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need a test file for `do_predict`.")
        
        # load
        if data_args.train_file.endswith(".csv"):
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    # get the column names and featuers
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    # get the text (or texts!!!) column name
    if data_args.task_name is not None:
        # get the sentence1_key and sentence2_key from the task name
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        # check if in our non_label_column_names we have sentence1 and sentence2
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        # check if in our non_label_column_names we have text and text2
        elif "text" in non_label_column_names:
            sentence1_key, sentence2_key = "text", None
        # assign default numbers
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # check if we are in a regression problem (stsb tasks or label is a float)
    if (data_args.task_name is not None and data_args.task_name=="stsb") or \
        (raw_datasets["train"].features["label"].dtype in ["float32", "float64"]):
        num_labels = 1    
        label_to_id = None
    else:
        # get the labels column name
        if data_args.label_column_name is not None:
            label_column_name = data_args.label_column_name
        elif "label" in column_names:
            label_column_name = "label"
        elif f"{data_args.task_name}_label" in column_names:
            label_column_name = f"{data_args.task_name}_label"
        else:
            label_column_name = column_names[-1]
        
        # get label vocabulary (train + validation)
        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels.add(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        label_list_train = get_label_list(raw_datasets["train"][label_column_name])
        label_list_val   = get_label_list(raw_datasets["validation"][label_column_name])
        label_list = list(set(label_list_train) | set(label_list_val))

        num_labels = len(label_list)
        label_to_id = {i: i for i in range(len(label_list))}

    # get padding and max sequence length
    if training_config_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if training_config_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({training_config_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(training_config_args.max_seq_length, tokenizer.model_max_length)
    
    # preprocess the dataset
    def preprocess_function(data):
        # tokenize the input text or texts
        args = (
            (data[sentence1_key],) if sentence2_key is None else (data[sentence1_key], data[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        
        # map labels to ids (note that we are outputting it in label*S* instead of label)
        if label_to_id is not None and "label" in data:
            result["labels"] = [(label_to_id[l] if l != -1 else -1) for l in data["label"]]
        
        # pad the labels to max seq length
        result["labels"]  = [[l] + [-100] * (max_seq_length - 1) for l in result["labels"]]
        
        # set the task ids
        result["task_ids"] = [task_id] * len(result["labels"])
        return result
    
    # tokenize using torch multiprocessing
    columns_to_drop = raw_datasets["train"].column_names    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched              = True,
            num_proc             = training_config_args.preprocessing_num_workers,
            load_from_cache_file = not training_config_args.overwrite_cache,
            remove_columns       = columns_to_drop,
            desc                 = "Running tokenizer on dataset (seq_class)",
        )
        
    task_info = Task(
        id          = task_id,
        name        = "seq_classification-id="+str(task_id),
        num_labels  = num_labels,
        type        = "seq_classification",
    )

    return tokenized_datasets["train"], tokenized_datasets["validation"], task_info

def load_datasets(tokenizer, model_args, data_args, training_config_args, training_args):
    idx = 0
    tasks = []
    for task in data_args:
        print("Loading dataset for",task)
        if task.task_type == "seq_classification":
            (train, val, task_info) = load_seq_classification_dataset(idx, tokenizer, model_args, task, training_config_args, training_args)
        
        elif task.task_type == "token_classification":
            (train, val, task_info) = load_token_classification_dataset(idx, tokenizer, model_args, task, training_config_args, training_args)
        
        # print train columns
        print(train.column_names)
        tasks.append({"train":train, "val":val, "task_info":task_info})
        idx += 1

    # merge datasets
    train_dataset_df    = tasks[0]["train"].to_pandas()
    validation_dataset  = [tasks[0]["val"]]
    tasks_info          = [tasks[0]["task_info"]]
    for i in range(1, len(tasks)):
        train_dataset_df    = pd.concat([train_dataset_df, tasks[i]["train"].to_pandas()], ignore_index=True)
        validation_dataset.append(tasks[i]["val"])
        tasks_info.append(tasks[i]["task_info"])

    train_dataset = datasets.Dataset.from_pandas(train_dataset_df)
    train_dataset.shuffle(seed=123)

    dataset = datasets.DatasetDict({"train": train_dataset, "validation": validation_dataset})
    
    return tasks_info, dataset

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    if preds.ndim == 2:
        # Token classification
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    elif preds.ndim == 3:
        # Sequence classification
        metric = load_metric("seqeval")

        predictions = np.argmax(preds, axis=2)

        true_predictions = [
            [f"tag-idx-{p}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, p.label_ids)
        ]
        true_labels = [
            [f"tag-idx-{l}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, p.label_ids)
        ]

        # Remove ignored index (special tokens)
        results = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    else:
        raise NotImplementedError()

def main(model_args, data_args, training_config_args, training_args):

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # load dataset/tasks and generate the mtl model
    tasks, raw_datasets = load_datasets(tokenizer, model_args, data_args, training_config_args, training_args)
    model = MultiTaskModel(model_args.encoder_name_or_path, tasks)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        
        if training_config_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(training_config_args.max_train_samples))

    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_datasets = raw_datasets["validation"]
        
        if training_config_args.max_eval_samples is not None:
            new_ds = []
            for ds in eval_datasets:
                new_ds.append(ds.select(range(training_config_args.max_eval_samples)))

            eval_datasets = new_ds

    # Log a few random samples from the training set:
    print(train_dataset)
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Initialize our Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset if training_args.do_train else None,
        compute_metrics = compute_metrics,
        tokenizer = tokenizer,
        data_collator = data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            training_config_args.max_train_samples
            if training_config_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:

        for eval_dataset, task in zip(eval_datasets, tasks):
            logger.info(f"*** Evaluate {task} ***")
            data_collator = None
            if task.type == "token_classification":
                data_collator = DataCollatorForTokenClassification(
                    tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
                )
            else:
                if training_config_args.pad_to_max_length:
                    data_collator = default_data_collator
                elif training_args.fp16:
                    data_collator = DataCollatorWithPadding(
                        tokenizer, pad_to_multiple_of=8
                    )
                else:
                    data_collator = None

            trainer.data_collator = data_collator
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                training_config_args.max_eval_samples
                if training_config_args.max_eval_samples is not None
                else len(eval_datasets)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_datasets))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    model_args = ModelArguments(encoder_name_or_path="bert-base-cased")
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        output_dir="./tmp/test",
        learning_rate=2e-5,
        num_train_epochs=3,
        overwrite_output_dir=True,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False
    )
    # NER Tags
    data_args_1 = DataTrainingArguments(
                        task_type="token_classification", 
                        dataset_name="conll2003",
                        text_column_name="tokens",
                        label_column_name="ner_tags")
    # Dependency head prediction
    data_args_2 = DataTrainingArguments(
                        task_type="token_classification", 
                        dataset_name="universal_dependencies", 
                        dataset_config_name="en_ewt",
                        text_column_name="tokens",
                        label_column_name="head")
    # POS Tags
    data_args_3 = DataTrainingArguments(
                        task_type="token_classification", 
                        dataset_name="universal_dependencies", 
                        dataset_config_name="en_ewt",
                        text_column_name="tokens",
                        label_column_name="upos")
    # Sentiment
    data_args_4 = DataTrainingArguments(
                        task_type="seq_classification",
                        dataset_name="tweet_eval",
                        dataset_config_name="emotion",
                        text_column_name="text",
                        label_column_name="label")
    

    training_config_args = TrainingConfigArguments(max_seq_length=128)
    main(model_args, [data_args_1, data_args_2, data_args_3, data_args_4], training_config_args, training_args)