from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import evaluate
from transformers import EvalPrediction
from torch import nn
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.data.data_collator import InputDataClass
from types import MappingProxyType
from frozendict import frozendict
from .heads.token_classification_head import TokenClassificationHead
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import transformers
from transformers.trainer_utils import EvalLoopOutput

class DataLoaderWithTaskname:
    def __init__(self, task_name, data_loader):
        self.task = task_name
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            yield batch

class NLPDataCollator:
    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        features = [{k:v for k,v in x.items() if k!='task_ids'} for x in features]
        return features
    
class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict, p=1):
        self.dataloader_dict = dataloader_dict
        N = max([len(x)**(1-p) for x in dataloader_dict.values()])
        
        f_p = lambda x: int(N*x**p)

        self.num_batches_dict = {
            task_name: f_p(len(dataloader))
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            f_p(len(dataloader.dataset)) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }

        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, tasks):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        tokenizer_kwargs = frozendict(padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name_or_path, **tokenizer_kwargs)
        self.output_heads = nn.ModuleDict()
        
        for task in tasks:
            task.set_tokenizer(self.tokenizer)
            for subtask in task.y:
                decoder = self._create_output_head(
                    self.encoder.config.hidden_size, 
                    task.task_type, 
                    task.num_labels[subtask]
                )
                
                self.output_heads[subtask] = decoder

        self.processed_tasks = self.preprocess_tasks(tasks, self.tokenizer)
        self.label_names = {task.name: task.label_names for task in tasks}
        self.train_dataset = {self.processed_tasks[task.name]['train'] for task in tasks}
        self.eval_dataset = {self.processed_tasks[task.name]['validation'] for task in tasks}
    
    def preprocess_tasks(self, tasks, tokenizer):      
        features_dict = {}
        for i, task in enumerate(tasks):
            print("[*] Model preprocessing task", task.name)
            
            if hasattr(task, 'processed_features') and tokenizer == task.tokenizer:
                features_dict[task.name] = task.processed_features
                continue
            
            for split in task.dataset:
                task.index = task.dataset[split].index = i
            
            features_dict[task.name] = {}
            for phase, phase_dataset in task.dataset.items():
                phase_dataset.index = i

                features_dict[task.name][phase] = phase_dataset.map(
                    task.preprocess_function, 
                    batched = True,
                    batch_size = 8,
                    load_from_cache_file = True
                )
        return features_dict
    
    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task_type, n_labels):
        if task_type == "TokenClassification":
            print("[*] Creating TokenClassification head with", n_labels, "labels")
            return TokenClassificationHead(encoder_hidden_size, n_labels)
        else:
            raise NotImplementedError()

    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None,
            head_mask = None, inputs_embeds = None, labels = None, task_ids = None, **kwargs):
            print("FORWARD MODEL FORWARD MODEL FORWARD MODEL")
            # compute the transformer output
            # this is never called?
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            sequence_output, pooled_output = outputs[:2]

            print("3) Transformer has been forwarded")
            unique_task_ids_list = torch.unique(task_ids).tolist()

            loss_list = []
            logits = None
            # print("Computing loss...")
            # print("task_ids", task_ids)
            print("==> Computing loss for the following tasks:")
            print("==>", unique_task_ids_list)
            for unique_task_id in unique_task_ids_list:
                print("Task_id =",unique_task_id)
                ptc_train = self.processed_tasks['train']
                target_cols = [col for col in ptc_train.features if col.startswith("target_")]
                print("target_cols =", target_cols)

                # for tc in target_cols:
                #     print("Target Column =",tc)
                #     print("Labels =",labels)
                #     logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                #         sequence_output[task_id_filter],
                #         pooled_output[task_id_filter],
                #         labels = None if labels is None else labels[task_id_filter],
                #         attention_mask=attention_mask[task_id_filter],
                #     )

                #     if labels is not None:
                #         loss_list.append(task_loss)

            # Loss averaged over all tasks
            outputs = (logits, outputs[2:])
            if loss_list:
                loss = torch.stack(loss_list)
                outputs = (loss.mean(),) + outputs

            return outputs


class MultiTaskTrainer(transformers.Trainer):
    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.p = 1
        self.processed_tasks = self.model.processed_tasks
        self.label_names = self.model.label_names
        self.train_dataset = {
            task: dataset["train"]
            for task, dataset in self.processed_tasks.items()
        }
        self.eval_dataset = {
            task: dataset["validation"]
            for task, dataset in self.processed_tasks.items()
        }
        self.eval_dataset = MappingProxyType(self.eval_dataset)
        self.tokenizer = self.model.tokenizer
        self.pretrained_transformer = self.model.encoder
        self.device = self.pretrained_transformer.device
        self.data_collator = NLPDataCollator(tasks)
        
        print("[*] Init multitask trainer with tasks:", self.processed_tasks)
        print("[*] Label names are:", self.label_names)
        print("[*] Heads are:", self.model.output_heads)
           
    def get_single_train_dataloader(self, task_name, train_dataset):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (SequentialSampler(train_dataset) if self.args.local_rank == -1 else DistributedSampler(train_dataset))
        data_loader = DataLoaderWithTaskname(
            task_name = task_name,
            data_loader = DataLoader(
                train_dataset,
                batch_size = self.args.train_batch_size,
                shuffle = False,
                sampler = train_sampler,
                collate_fn = self.data_collator.__call__,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }, p = self.p,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in (
                    eval_dataset if eval_dataset else self.eval_dataset
                ).items()
            }
        )

    def evaluation_loop(self, dataloader: DataLoader, description: str, 
                        prediction_loss_only: bool | None = None, ignore_keys: List[str] | None = None, 
                        metric_key_prefix: str = "eval_") -> EvalLoopOutput:
          
        # this is useless?
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        for step, inputs in enumerate(dataloader):            
            loss, preds, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            
            for task, label_names in self.label_names.items():
                preds_task = preds[task]
                labels_task = labels[task]
                
                for label_name, labels_values in label_names.items():
                    preds_tl  = preds_task[label_name]
                    labels_tl = labels_task[label_name]
                    
                    eval_pred = EvalPrediction(
                                predictions = preds_tl, 
                                label_ids   = labels_tl, 
                                inputs      = inputs)

                    # compute metrics foreach head using the corresponding task eval_function
                    # i copied the function from the task-specific class to this one
                    metrics = self.compute_metrics_token_classification(eval_pred, label_name)
                    metrics_eval = {}
                    for metric in metrics.items():
                        metrics_eval[metric_key_prefix + "_" + metric[0]] = metric[1]

        return EvalLoopOutput(predictions=preds_tl, label_ids=labels_tl, metrics=metrics_eval, num_samples=len(self.eval_dataset))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=[]):
        if ignore_keys is None:
            ignore_keys = []

        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.mean().detach()
        
        logits_dict = {}
        labels_dict = {}
        for task_name, label_names in self.label_names.items():
            logits_dict[task_name] = {}
            labels_dict[task_name] = {}
            for label_name in label_names:
                logits_dict[task_name][label_name] = outputs[label_name]
                logits_dict[task_name][label_name] = np.argmax(outputs[label_name].detach().cpu().numpy(), axis=2)
                target_labels = []
                for i in inputs:
                    target_labels.append(i[label_name])
                labels_dict[task_name][label_name] = torch.tensor(target_labels)
        
        return (loss, logits_dict, labels_dict)

    def compute_metrics_token_classification(self, eval_pred, label_names):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
        # for task in tasks
        for task_name in self.label_names:
            task_name = self.label_names['naive_absolute_n_commons']
            true_labels = [
                [task_name[label_names][int(l)] for l in label if l != -100] for label in labels
            ]
            
            true_predictions = [
                [task_name[label_names][p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            metric = evaluate.load("seqeval")
            all_metrics = metric.compute(
                predictions = true_predictions, 
                references = true_labels
            )
            
            meta = {"name": task_name, "size": len(predictions), "index": 0}
            print(all_metrics)
            metrics = {k.replace("overall_",""):v for k,v in all_metrics.items() if "overall" in k}
        
        print(metrics)
        return {**metrics, **meta}      

    def compute_loss(self, model, inputs, return_outputs=False):
        keys = inputs[0].keys()

        input_ids = torch.tensor([i['input_ids'] for i in inputs], device=self.args.device) if 'input_ids' in keys else None
        attention_mask = torch.tensor([i['attention_mask'] for i in inputs], device=self.args.device) if 'attention_mask' in keys else None        
        token_type_ids = torch.tensor([i['token_type_ids'] for i in inputs], device=self.args.device) if 'token_type_ids' in keys else None        
        position_ids = torch.tensor([i['position_ids'] for i in inputs], device=self.args.device) if 'position_ids' in keys else None        
        head_mask = torch.tensor([i['head_mask'] for i in inputs], device=self.args.device) if 'head_mask' in keys else None        
        inputs_embeds = torch.tensor([i['inputs_embeds'] for i in inputs], device=self.args.device) if 'inputs_embeds' in keys else None
        
        outputs = self.pretrained_transformer(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]
        loss_list = []
        logits_list = {}
        
        
        for i, head in enumerate(self.model.output_heads.values()):
            labels_name = f"target_{i+1}" if i > 0 else "target"
            labels_i = torch.tensor([i[labels_name] for i in inputs], device=self.args.device)
            logits, loss = head(sequence_output, pooled_output, labels=labels_i, attention_mask=attention_mask)
            loss_list.append(loss)
            logits_list[labels_name] = logits
        loss = torch.stack(loss_list)
        loss = torch.mean(loss)
        return (loss, logits_list) if return_outputs else loss
