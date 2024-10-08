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

        # load pretrained model and send to cuda device 
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.to(device)
        
        tokenizer_kwargs = frozendict(padding="max_length", truncation=True, return_tensors="pt")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        print("[*] Loaded model with tokenizer:", self.tokenizer)
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
        self.label_names     = {task.name: task.label_names for task in tasks}
        print("[*] Created model with label names =", self.label_names)

        self.train_dataset   = {task.name: self.processed_tasks[task.name]['train'] for task in tasks} if do_train else None
        self.eval_dataset    = {task.name: self.processed_tasks[task.name]['validation'] for task in tasks} if do_eval else None
        self.test_dataset    = {task.name: self.processed_tasks[task.name]['test'] for task in tasks} if do_test else None
       
        print("[*] Created model with train dset =", self.train_dataset)
        print("[*] Created model with eval dset =", self.eval_dataset)
        print("[*] Created model with test dset =", self.test_dataset)
        


    def preprocess_tasks(self, tasks, tokenizer):      
        features_dict = {}
        for i, task in enumerate(tasks):
            print("[*] Preprocessing task: ", task.name,"***")
            
            if hasattr(task, 'processed_features') and tokenizer == task.tokenizer:
                features_dict[task.name] = task.processed_features
                continue
            
            for split in task.dataset:
                task.index = task.dataset[split].index = i
            
            features_dict[task.name] = {}
            for phase, phase_dataset in task.dataset.items():
                phase_dataset.index = i

                # TODO: FIX THIS FIXED INT
                features_dict[task.name][phase] = phase_dataset.map(
                    task.preprocess_function, 
                    batched = True,
                    batch_size = 64,
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
        print(self.label_names)
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
        
        print("[*] Init HF-MTL trainer with tasks:", self.task_names)
        print("    Running on device:", self.device)
        print("    Evaluation strategy:", self.args.evaluation_strategy)
        print("    Logging strategy:", self.args.logging_strategy) 
        print("    Training:", self.args.do_train)
        print("    Evaluation:", self.args.do_eval)
        print("    Prediction:", self.args.do_predict)


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

        eval_loop = self.evaluation_loop
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
                num_samples = output.num_samples,
                num_steps = math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        print("[*] Prediction results:")
        for m in output.metrics:
            print('\t',m, output.metrics[m])



        decoded_predicted_labels = {}        
        label_names = self.label_names
        print("Label names:", label_names)
        for task in label_names:
            decoded_predicted_labels[task] = {}
            
            for target in label_names[task]:
                decoded_predicted_labels[task][target] = []
                current_task_labels_dictionary = self.label_names[task][target]
                for batch in output.predictions[task]:
                    for sentences in batch[target]:
                        for current_sentence in sentences:
                            current_sentence_decoded = []
                            for idx, l in enumerate(current_sentence):
                                current_sentence_decoded.append(current_task_labels_dictionary[int(l)])
                            decoded_predicted_labels[task][target].append(current_sentence_decoded)
        
        

        def restore_label_alignment(input_ids, label_ids):
            '''
            Restores the label_ids to match the input_ids by removing sub-word labels;
            as we are using roberta tokenizer, we need to remove the labels corresponding to sub-words
            (i.e. labels with corresponding de-tokenized input_ids starting with 'Ġ')
            '''
            new_labels = []
            restored_text = []
            for i, token in enumerate(input_ids):
                if token.startswith("Ġ") or i==0:
                    new_labels.append(label_ids[i])
                    restored_text.append(token[1:])
                else:
                    restored_text[-1] = restored_text[-1] + token
            return new_labels, restored_text
        
        fixed_labels = {}
        for task in self.task_names:
            test_dataset = test_dataset[task]
            task_outputs = decoded_predicted_labels[task]
            for target in task_outputs.keys():
                task_target_outputs = task_outputs[target]
                for i, sentence in enumerate(task_target_outputs):
                    detokenized_tokens_i = self.tokenizer.convert_ids_to_tokens(test_dataset['input_ids'][i], skip_special_tokens=True)
                    fixed_label_i, fixed_text_i = restore_label_alignment(detokenized_tokens_i, sentence)
                    if len(fixed_label_i) != len(fixed_text_i):
                        print("Input text:", fixed_text_i)
                        print("Fixed label:", fixed_label_i)
                        exit()
                    if task not in fixed_labels:
                        fixed_labels[task] = {}
                    if target not in fixed_labels[task]:
                        fixed_labels[task][target] = []
                    fixed_labels[task][target].append(fixed_label_i)                  

        # print a prediction from each sample
        return PredictionOutput(predictions=output.predictions, label_ids=fixed_labels, metrics=output.metrics)

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
            print("Evaluating step:", step)
            print("\tInputs:", len(inputs))
            loss, preds, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # iterate over all tasks
            for task in self.task_names:
                print("\tEvaluating task:", task)
                predictions_current_task = preds[task]
                real_current_task = labels[task]
                
                preds_i = {label_name: [] for label_name in self.label_names[task]}
                lbls_i  = {label_name: [] for label_name in self.label_names[task]}
                
                # iterate over all task targets
                for label_name, labels_values in self.label_names[task].items():
                    preds_current_target  = predictions_current_task[label_name]
                    real_current_target   = real_current_task[label_name]

                    # print tuple of predictions and labels
                    print("\tPredictions:", len(preds_current_target))
                    print("\tLabels:", len(real_current_target))

                    # pred current target is a list
                    # real current target is a tensor
                    eval_pred = EvalPrediction(
                                predictions = preds_current_target, 
                                label_ids   = real_current_target, 
                                inputs      = inputs)

                    metrics = self.compute_metrics_token_classification(eval_pred, 
                                                                        task, 
                                                                        label_name)
                    print("\tEvaluation results:")
                    for m in metrics:
                        print("\t\t", m, metrics[m])

                    metrics_eval = {}
                    for metric in metrics.items():
                        metrics_eval[metric_key_prefix + "_" + metric[0]] = metric[1]

                    if return_preds:
                        for sequence in real_current_target:
                            current = []
                            for l in sequence:
                                if l == -100:
                                    continue
                                l_i = labels_values[int(l)] 
                                current.append(l_i)
                            lbls_i[label_name].append(current)                                

                    preds_i[label_name].append(preds_current_target)
                    lbls_i[label_name].append(real_current_target)

                
                predictions[task].append(preds_i)
                label_ids[task].append(lbls_i)
                print("    Done evaluating task:", task)
        
        return EvalLoopOutput(predictions=predictions, label_ids=label_ids, metrics=metrics_eval, num_samples = len(dataloader))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=[]):
        if ignore_keys is None:
            ignore_keys = []

        # the logits are the predictions of the model
        # compute mean loss over all tasks
        loss, logits_list = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.mean().detach()
        
        logits_dict = {}
        labels_dict = {}

        for task_name, label_names in self.label_names.items():
            logits_dict[task_name] = {}
            labels_dict[task_name] = {}
        
            for label_name in label_names:
                logits_dict[task_name][label_name] = logits_list[label_name]
                logits_dict[task_name][label_name] = np.argmax(logits_list[label_name]
                                                               .detach().cpu().numpy(), axis=2)
                target_labels = [i[label_name] for i in inputs]
                labels_dict[task_name][label_name] = torch.tensor(target_labels)
                
        # labels dict shaped as {task_name: {label_name: [labels]}}
        # logits dict shaped as {task_name: {label_name: [logits]}}
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
            logits, loss = head(sequence_output, pooled_output, labels=labels_i, attention_mask = attention_mask)
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
        
        # print("Comparing true predictions with true labels")
        # print([(len(x), len(y)) for x,y in zip(true_predictions, true_labels)])
        # # print a side by side sample comparison
        # print("Sample comparison:")
        # for i in range(0,5):
        #     print("-"*50)
        #     print("Predictions:", true_predictions[i])
        #     print("True Labels:", true_labels[i])

        metric = evaluate.load("seqeval")

        all_metrics = metric.compute(
            predictions = true_predictions, 
            references = true_labels
        )
        
        meta    = {"task_name": f"{task}_{label_names}", "size": len(predictions), "index": 0}
        metrics = {k.replace("overall_",""):v for k,v in all_metrics.items() if "overall" in k}
        return {**metrics, **meta}      


