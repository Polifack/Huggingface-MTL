from typing import Any, Dict, List, Optional, Tuple, Union
import time
import torch
import numpy as np
import evaluate
import math
from transformers import EvalPrediction
from torch import nn
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.data.data_collator import InputDataClass
from types import MappingProxyType
from frozendict import frozendict
from .heads.token_classification_head import TokenClassificationHead
from .heads.sequence_classification_head import SequenceClassificationHead
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import transformers
from transformers.trainer_utils import EvalLoopOutput, PredictionOutput, speed_metrics

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
    def __init__(self, model_name, do_train, do_eval, do_test, tasks):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        tokenizer_kwargs = frozendict(padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.output_heads = nn.ModuleDict()
        
        for task in tasks:
            task.set_tokenizer(self.tokenizer)
            for subtask in task.y:
                decoder = self._create_output_head(
                    self.encoder.config.hidden_size, 
                    task.task_type, 
                    task.num_labels[subtask]
                )
                
                print("=====> subtask = ", subtask)
                self.output_heads[subtask] = decoder

        self.processed_tasks = self.preprocess_tasks(tasks, self.tokenizer)
        self.label_names     = {task.name: task.label_names for task in tasks}

        self.train_dataset   = {task.name: self.processed_tasks[task.name]['train'] for task in tasks} if do_train else None
        self.eval_dataset    = {task.name: self.processed_tasks[task.name]['validation'] for task in tasks} if do_eval else None
        self.test_dataset    = {task.name: self.processed_tasks[task.name]['test'] for task in tasks} if do_test else None

        print("Created model with train dset =", self.train_dataset, type(self.train_dataset))
        print("Created model with eval dset =", self.eval_dataset, type(self.eval_dataset))
        print("Created model with test dset =", self.test_dataset, type(self.test_dataset))
    
    def preprocess_tasks(self, tasks, tokenizer):      
        features_dict = {}
        for i, task in enumerate(tasks):
            print("[*] Model preprocessing task: ", task.name,"***")
            
            if hasattr(task, 'processed_features') and tokenizer == task.tokenizer:
                features_dict[task.name] = task.processed_features
                continue
            
            for split in task.dataset:
                task.index = task.dataset[split].index = i
            
            features_dict[task.name] = {}
            for phase, phase_dataset in task.dataset.items():
                phase_dataset.index = i

                # fix fixed batch size
                features_dict[task.name][phase] = phase_dataset.map(
                    task.preprocess_function, 
                    batched = True,
                    batch_size = 8,
                    load_from_cache_file = True
                )

            print("[*] Finished preprocessing task: ", task.name,"***")

        return features_dict
    
    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task_type, n_labels):
        if task_type == "TokenClassification":
            print("[*] Creating TokenClassification head with", n_labels, "labels")
            return TokenClassificationHead(encoder_hidden_size, n_labels)
        elif task_type == "SequenceClassification":
            print("[*] Creating SequenceClassification head with", n_labels, "labels")
            return SequenceClassificationHead(encoder_hidden_size, n_labels)

        else:
            raise NotImplementedError()

class MultiTaskTrainer(transformers.Trainer):
    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.p = 1
        self.processed_tasks = self.model.processed_tasks
        self.n_tasks = len(self.processed_tasks)
        
        self.label_names = self.model.label_names
        self.task_names = list(self.processed_tasks.keys())
        
        self.train_dataset = {task: dataset["train"] for task, dataset in self.processed_tasks.items()} if self.args.do_train else None
    
        self.eval_dataset = {task: dataset["validation"] for task, dataset in self.processed_tasks.items()} if self.args.do_eval else None
        self.eval_dataset = MappingProxyType(self.eval_dataset) if self.eval_dataset else None
        
        self.test_dataset = {task: dataset["test"] for task, dataset in self.processed_tasks.items()} if self.args.do_predict else None
        self.test_dataset = MappingProxyType(self.test_dataset) if self.test_dataset else None
        
        self.tokenizer = self.model.tokenizer
        self.pretrained_transformer = self.model.encoder
        self.device = self.pretrained_transformer.device
        self.data_collator = NLPDataCollator(tasks)
        
        print("[*] Init multitask trainer with tasks:", self.task_names)
        print("[*] Label names are:", self.label_names)

    def get_single_train_dataloader(self, task_name, train_dataset):
        if self.train_dataset is None and self.args.do_train:
            raise ValueError("[*] Error Trainer: training requires a train_dataset.")

        train_sampler = SequentialSampler(train_dataset)# if self.args.local_rank == -1 else DistributedSampler(train_dataset))
        data_loader = DataLoaderWithTaskname(
            task_name = task_name,
            data_loader = DataLoader(
                train_dataset,
                batch_size = self.bsize,
                # shuffle = not self.args.do_predict, # only shuffle during training
                shuffle = False,
                sampler = train_sampler,
                collate_fn = self.data_collator.__call__,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        self.bsize = self.args.per_device_train_batch_size
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }, p = self.p,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        self.bsize = self.args.per_device_eval_batch_size
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in (
                    eval_dataset if eval_dataset else self.eval_dataset
                ).items()
            }
        )

    def get_test_dataloader(self, test_dataset=None):
        self.bsize = self.args.per_device_eval_batch_size
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in (
                    test_dataset if test_dataset else self.test_dataset
                ).items()
            }
        )

    def predict(self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"):
        print("*********************************")
        print("***     Prediction Loop       ***")
        print("*********************************")

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, 
            description="Prediction", 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix,
            return_preds=True
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
            
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        for m in output.metrics:
            print(m, output.metrics[m])

        # restore labels
        # for t in self.task_names:
        #     label_names = self.label_names[t]
        #     for sentence in label_names:
        #         print(sentence)

        # error here
        lbls_idx = 0
        for step, inputs in enumerate(test_dataloader):
            print("Step:", step,"/",len(test_dataloader))
            for seq in inputs:
                print(seq)
                for task in self.task_names:
                    label_names = self.label_names[task]
                    print(label_names[lbls_idx])
                lbls_idx += 1

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(self, 
                        dataloader: DataLoader, 
                        description: str, 
                        prediction_loss_only: bool | None = None, 
                        ignore_keys: List[str] | None = None, 
                        metric_key_prefix: str = "eval_",
                        return_preds: bool = False) -> EvalLoopOutput:
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        predictions = {task: [] for task in self.task_names}
        label_ids   = {task: [] for task in self.task_names}
        for step, inputs in enumerate(dataloader):            
            print("Step:", step,"/",len(dataloader))
            
            loss, preds, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            for task in self.task_names:
                print("    Evaluating task:", task)
                preds_task = preds[task]
                labels_task = labels[task]
                
                preds_i = {label_name: [] for label_name in self.label_names[task]}
                lbls_i = {label_name: [] for label_name in self.label_names[task]}
                
                for label_name, labels_values in self.label_names[task].items():
                    print("       Evaluating target:", label_name)
                    preds_tl  = preds_task[label_name]
                    labels_tl = labels_task[label_name]
                    
                    eval_pred = EvalPrediction(
                                predictions = preds_tl, 
                                label_ids   = labels_tl, 
                                inputs      = inputs)

                    metrics = self.compute_metrics_token_classification(eval_pred, task, label_name)
                    print("       Evaluation results:",metrics)
                    metrics_eval = {}
                    for metric in metrics.items():
                        metrics_eval[metric_key_prefix + "_" + metric[0]] = metric[1]

                    if return_preds:
                        for sequence in labels_tl:
                            current = []
                            for l in sequence:
                                if l == -100:
                                    continue
                                l_i = labels_values[int(l)] 
                                current.append(l_i)
                            lbls_i[label_name].append(current)                                

                    preds_i[label_name].append(preds_tl)
                    lbls_i[label_name].append(labels_tl)

                
                predictions[task].append(preds_i)
                label_ids[task].append(lbls_i)

        return EvalLoopOutput(predictions=predictions, label_ids=label_ids, metrics=metrics_eval, num_samples = len(dataloader))

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
        
        i = 0
        for key, head in (self.model.output_heads.items()):
            if key not in keys:
                continue
            labels_name = key
            labels_i = torch.tensor([i[labels_name] for i in inputs], device=self.args.device)
            logits, loss = head(sequence_output, pooled_output, labels=labels_i, attention_mask=attention_mask)
            loss_list.append(loss)
            logits_list[labels_name] = logits
        loss = torch.stack(loss_list)
        loss = torch.mean(loss)
        
        return (loss, logits_list) if return_outputs else loss

    def compute_metrics_token_classification(self, eval_pred, task, label_names):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
        # for task in tasks
        label_values = self.label_names[task][label_names]
        true_labels = [
            [label_values[int(l)] for l in label if l != -100] for label in labels
        ]
        
        true_predictions = [
            [label_values[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        metric = evaluate.load("seqeval")
        #metric = evaluate.load('./evaluation')

        all_metrics = metric.compute(
            predictions = true_predictions, 
            references = true_labels
        )
        
        meta    = {"task_name": f"{task}_{label_names}", "size": len(predictions), "index": 0}
        metrics = {k.replace("overall_",""):v for k,v in all_metrics.items() if "overall" in k}
        return {**metrics, **meta}      


