<<<<<<< Updated upstream
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from typing import List, Union, Dict
from transformers.trainer import EvalLoopOutput
from transformers import (
    EncoderDecoderModel,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoModelForTokenClassification,
)
from easydict import EasyDict as edict
import funcy as fc
import copy
import logging
from types import MappingProxyType
from .tasks.sequence_classification import SequenceClassification
from .utils import to_dict, shallow_copy_A_to_B, deep_copy_cache, normalize_label, NoTqdm, search_module
from transformers import AutoTokenizer
import magicattr
import gc
import random
from tqdm.auto import tqdm
from transformers import pipeline

class Adapter(transformers.PreTrainedModel):
    config_class = transformers.PretrainedConfig
    def __init__(self, config, classifiers=None, Z=None, labels_list=[]):
        super().__init__(config)    
        self.Z= torch.nn.Embedding(len(config.classifiers_size),config.hidden_size).weight if Z==None else Z
        self.classifiers=torch.nn.ModuleList(
            [torch.nn.Linear(config.hidden_size,size) for size in config.classifiers_size]
        ) if classifiers==None else classifiers
        self.config=self.config.from_dict(
            {**self.config.to_dict(),
            'labels_list':labels_list}
        )
    def adapt_model_to_task(self, model, task_name):
        task_index=self.config.tasks.index(task_name)
        setattr(model,search_module(model,'linear',mode='class')[-1], self.classifiers[task_index])
        return model
    def _init_weights(*args):
        pass 

class ConditionalLayerNorm(torch.nn.Module):
    def __init__(self, LN, Z_i, drop_probability=0.0):
        super().__init__()
        self.LN = LN
        self.Z_i = Z_i
        size,task_emb_size =len(LN.bias), len(Z_i)
        self.L1 = torch.nn.Linear(task_emb_size, size*2)
        self.L1.apply(lambda x: self.weight_init(x, k=size))
        self.gates = torch.nn.Parameter(torch.ones(2))
        self.sigmoid = torch.nn.Sigmoid()
        self.drop_probability=drop_probability

    @classmethod
    def weight_init(cls, m,std=1e-3,k=1):
        std=std/(k**.5)
        m.weight.data.normal_(0.0, std).clamp_(-2*std,2*std)
        m.bias.data.zero_()
        
    def forward(self, inputs):
        gates = self.sigmoid(self.gates)
        if random.random()<self.drop_probability:
            a,b = self.LN.weight, self.LN.bias
        else:
            c,d=self.L1(self.Z_i).chunk(2,dim=-1)
            a = gates[0]*c + self.LN.weight
            b = gates[1]*d + self.LN.bias
        return torch.nn.functional.layer_norm(inputs, self.LN.normalized_shape, a,b, eps=self.LN.eps)

class CLSEmbedding(nn.Module):
    def __init__(self, Z_i, drop_probability=0.0):
        super().__init__()
        self.cls = Z_i
        self.drop_probability=drop_probability
    def forward(self, x):
        if random.random()>self.drop_probability:
            x[:, 0, :] = x[:, 0, :] + self.cls.to(x.device)
        return x

class NLPDataCollator:
    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(
        self, features: List[Union[InputDataClass, Dict]]
    ) -> Dict[str, torch.Tensor]:
        try:
            task_index = features[0]["task"].flatten()[0].item()
        except:
            print("features:",features)
            task_index = features[-1]["task"].flatten()[0].item()
            
        features = [{k:v for k,v in x.items() if k!='task'} for x in features]
        collated = self.tasks[task_index].data_collator.__call__(features)
        collated['task']=torch.tensor([task_index])
        return collated
=======
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import transformers
from transformers.trainer_utils import EvalLoopOutput
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
=======
class NLPDataCollator:
    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        features = [{k:v for k,v in x.items() if k!='task_ids'} for x in features]
        return features
    
>>>>>>> Stashed changes
class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict, p=1):
        self.dataloader_dict = dataloader_dict
        N = max([len(x)**(1-p) for x in dataloader_dict.values()])
<<<<<<< Updated upstream
=======
        
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
def add_cls(model, Z_i, drop_probability=0.0):
    """model is a standard HF Transformer"""
    emb_name, emb_module = [(name,module) for name,module in model.named_modules() if isinstance(module,torch.nn.Embedding)][0]
    magicattr.set(model, emb_name,
        nn.Sequential(emb_module, 
        CLSEmbedding(Z_i,
        drop_probability=drop_probability
    )))

def remove_cls(model):
    model=copy.copy(model)
    cls_embeddings = [(name,module) for name,module in model.named_modules() if isinstance(module,torch.nn.Sequential)
        and isinstance(module[-1], CLSEmbedding)]
    if cls_embeddings:
        emb_name, emb_module = cls_embeddings[0]
        magicattr.set(model, emb_name, emb_module[0])
    return model

def add_cln(model,Z_i,drop_probability=0.0):
    for ln in search_module(model, 'layernorm'):
        magicattr.set(model,ln, 
        ConditionalLayerNorm(magicattr.get(model,ln), Z_i, drop_probability=drop_probability)
        )

def last_linear(classifier):
    L = list([m for m in classifier.modules() if type(m)==torch.nn.Linear])[-1]
    return L



class Model(transformers.PreTrainedModel):
    def __init__(self, tasks, args):
        super().__init__(transformers.PretrainedConfig())
        
        self.models = {}
        self.shared_encoder = None
        
        # get tasks
        self.task_names = [t.name for t in tasks]
        self.task_labels_list = [t.get_labels() for t in tasks]
        
        # get model parameters
        self.batch_truncation = args.get('batch_truncation', False)
        self.add_cls = args.get('add_cls', True)
        self.add_cln = args.get('add_cln', False)
        self.drop_probability = args.get('drop_probability', 0.1)
        
        task_models_list = []
        
        for i, task in enumerate(tasks):
            print("[*] Found task",i,"=>",task.name)
            model_type = eval(f"AutoModelFor{task.task_type}")

            nl    = {a: getattr(task, a) for a in ('num_labels', 'problem_type') if hasattr(task, a)}
            
            # this also does not work
            # model = deep_copy_cache(model_type.from_pretrained)(args.model_name, ignore_mismatched_sizes=True, load_in_8bit=True, device_map='auto', **nl)
            
            model = deep_copy_cache(model_type.from_pretrained)(args.model_name, ignore_mismatched_sizes=True, **nl)
            
            labels = getattr(task.dataset["train"].features[task.y], "names", None)
            key    = tuple([normalize_label(x) for x in labels]) if labels else None

            if key and key not in self.models:
                self.models[key] = model 
            if key and key in self.models:
                last_linear(model.classifier).weight = last_linear(self.models[key].classifier).weight

            model.auto = getattr(model, self.get_encoder_attr_name(model))

            if self.shared_encoder is None:
                self.shared_encoder = model.auto
            else:
                shallow_copy_A_to_B(self.shared_encoder, model.auto)
            
            task_models_list += [model]
            model.i = i

        self.task_models_list = nn.ModuleList(task_models_list)

        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        
        
        self.Z = nn.parameter.Parameter(
            torch.zeros(len(tasks),
            self.shared_encoder.config.hidden_size, device=device),
            requires_grad=len(tasks)>1
            )

        for i, task in enumerate(tasks):
            m_i = self.task_models_list[i]
            if self.add_cls:
                add_cls(m_i,self.Z[i],drop_probability=self.drop_probability)
            if self.add_cln:
                add_cln(m_i,self.Z[i][::8],
                drop_probability=self.drop_probability)
        torch.cuda.empty_cache()
        gc.collect()

    def set_encoder(self,encoder):
        for model in self.task_models_list:
            shallow_copy_A_to_B(encoder, getattr(model, self.get_encoder_attr_name(model)))

    @classmethod
    def get_encoder_attr_name(cls, model):
        if hasattr(model,'model'):
            return 'model'
        if hasattr(model, "encoder"):
            return "encoder"
        else:
            return model.config.model_type.split('-')[0]

    def batch_unpad(self,kwargs,task_index):
        """Remove excess padding (improves speed)"""

        batch_max_size=kwargs['attention_mask'].sum(axis=1).max().item()
        kwargs['attention_mask']=kwargs['attention_mask'][:,:batch_max_size].contiguous() 
        kwargs['input_ids']=kwargs['input_ids'][:,:batch_max_size].contiguous() 
        
        if len(kwargs['labels'].shape)>1 \
            and self.task_models_list[task_index].config.problem_type!="multi_label_classification":
            kwargs['labels']=kwargs['labels'][:,:batch_max_size].contiguous() 
        return kwargs

    def forward(self, task, **kwargs):
        task_index = task[0].item()
        if self.batch_truncation:
            kwargs = self.batch_unpad(kwargs, task_index)
        y = self.task_models_list[task_index](**kwargs)
        return y

    def factorize(self, task_index=0, tasks=[]):
        m_i = self.task_models_list[task_index]

        classifiers = torch.nn.ModuleList([a.classifier for a in self.task_models_list])
        id2label=dict(enumerate(self.task_labels_list[task_index]))
        label2id = {str(v):k for k,v in id2label.items()}

        m_i.config = m_i.config.from_dict(
            {**m_i.config.to_dict(),
            'classifiers_size': [c.out_features for c in classifiers],
            'tasks': (tasks if tasks else self.task_names),
            'label2id':label2id,'id2label':id2label
            })
        adapter=Adapter(m_i.config, classifiers, self.Z, self.task_labels_list)

        if not hasattr(m_i,"factorized"):
            if hasattr(m_i,'auto'):
                del m_i.auto    
            m_i=remove_cls(m_i)
            m_i.factorized=True

        return m_i, adapter


class Trainer(transformers.Trainer):
    def __init__(self, model, tasks, hparams, tokenizer=None, *args, **kwargs):
        class default:
            output_dir = "./models/multitask_model"
            overwrite_output_dir = True
            
            do_train = True
            per_device_train_batch_size = 16
            
            do_eval  = True
            per_device_eval_batch_size = 16
            
            evaluation_strategy = "steps"
            eval_steps = 64

            logging_strategy = "epoch"
            logging_steps = 64
            
            save_steps = 1000000
            
            label_names = ["labels"]
            include_inputs_for_metrics = True
        
        ## Load pre-trained transformer model        
        default, hparams_dict = to_dict(default), to_dict(hparams)
        self.p = hparams_dict.get('p', 0)
        self.num_proc = hparams_dict.get('num_proc', None)
        self.batched = hparams_dict.get('batched', False)

        trainer_args = transformers.TrainingArguments(
            **{**default, **fc.project(hparams_dict, dir(transformers.TrainingArguments))},
        )

        ## Set the number of gpus (quick fix for now)
        trainer_args._n_gpu = 1
        self.n_gpus = trainer_args._n_gpu
    
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(hparams_dict["model_name"])
        
        if 'max_length' in hparams_dict:
            for t in tasks:
                t.tokenizer_kwargs['max_length'] = hparams_dict['max_length']
        
        super().__init__(
            model,
            trainer_args,
            tokenizer = tokenizer,
            compute_metrics = SequenceClassification.compute_metrics
        )

        self.per_device_train_batch_size = self.args.train_batch_size
        self.data_collator = NLPDataCollator(tasks)

        ## Load tastks
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.processed_tasks = self.preprocess_tasks(tasks, self.tokenizer)
        
=======

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
>>>>>>> Stashed changes
        self.train_dataset = {
            task: dataset["train"]
            for task, dataset in self.processed_tasks.items()
        }
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
        self.eval_dataset = {
            task: dataset["validation"]
            for task, dataset in self.processed_tasks.items()
        }
<<<<<<< Updated upstream

        self.test_dataset = {
            task: dataset["test"]
            for task, dataset in self.processed_tasks.items()
        }
        
        # We prevent trainer from automatically evaluating on each dataset: transformers.Trainer recognizes 
        # eval_dataset instances of "dict" but we use a custom "evaluate" function so that we can use 
        # different metrics for each task
        self.eval_dataset = MappingProxyType(self.eval_dataset)

    @staticmethod
    def write_line(other, values):
        if other.inner_table is None:
            other.inner_table = [list(values.keys()), list(values.values())]
        else:
            columns = other.inner_table[0]
            for key in values.keys():
                if key not in columns:
                    columns.append(key)
            other.inner_table[0] = columns
            other.inner_table.append([values.get(c, np.nan) for c in columns])

    def evaluate(self, metric_key_prefix="eval", **kwargs):
        # logging
        try:
            i = [i for (i,c) in enumerate(self.callback_handler.callbacks) if 'NotebookProgress' in str(c)][0]
            self.callback_handler.callbacks[i].training_tracker.write_line = fc.partial(
                self.write_line, self.callback_handler.callbacks[i].training_tracker
            )
        except:
            logging.info('No training_tracker')
        
        
        outputs = []
        print("=>", kwargs)
        for i, task in enumerate(self.tasks):
            print("[*] Evaluating task", i, "=>", task.name)
            
            self.compute_metrics = task.compute_metrics
            eval_dataset =  dict([fc.nth(i, (self.eval_dataset if metric_key_prefix == "eval" else self.test_dataset).items())])
            output = transformers.Trainer.evaluate(self, eval_dataset = eval_dataset, metric_key_prefix = metric_key_prefix)
            
            if "Accuracy" not in output:
                output["Accuracy"] = np.nan
            outputs += [output]
        return fc.join(outputs) if metric_key_prefix!="test" else outputs

    def task_batch_size(self,task_name):
        if hasattr(task_name, 'num_choices'):            
            return max(1, self.args.train_batch_size // task_name.num_choices)
        else:
            return self.args.train_batch_size

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

=======
        self.eval_dataset = MappingProxyType(self.eval_dataset)
        self.tokenizer = self.model.tokenizer
        self.pretrained_transformer = self.model.encoder
        self.device = self.pretrained_transformer.device
        self.data_collator = NLPDataCollator(tasks)
        
        print("[*] Init multitask trainer with tasks:", self.processed_tasks)
        print("[*] Label names are:", self.label_names)
           
    def get_single_train_dataloader(self, task_name, train_dataset):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (SequentialSampler(train_dataset) if self.args.local_rank == -1 else DistributedSampler(train_dataset))
>>>>>>> Stashed changes
        data_loader = DataLoaderWithTaskname(
            task_name = task_name,
            data_loader = DataLoader(
                train_dataset,
<<<<<<< Updated upstream
                batch_size = self.task_batch_size(task_name),
=======
                batch_size = self.args.train_batch_size,
                shuffle = False,
>>>>>>> Stashed changes
                sampler = train_sampler,
                collate_fn = self.data_collator.__call__,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
<<<<<<< Updated upstream
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
=======
>>>>>>> Stashed changes
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }, p = self.p,
        )
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    def get_eval_dataloader(self, eval_dataset=None):
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in (
                    eval_dataset if eval_dataset else self.eval_dataset
                ).items()
            }
        )

<<<<<<< Updated upstream
    def get_test_dataloader(self, test_dataset=None):
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in (
                    test_dataset if test_dataset else self.test_dataset
                ).items()
            }
        )

    def pipeline(self,task_index=0):
        m,_ = self.model.factorize(task_index=task_index)
        return pipeline("text-classification",model=m,tokenizer=self.tokenizer,
                device=m.device,padding=True)

    def save_model(self,output_dir,task_index=0,**kwargs):
        model, adapter = self.model.factorize(task_index=task_index)
        model.save_pretrained(output_dir)
        adapter.save_pretrained(f"{output_dir}-adapter")
    
    def push_to_hub(self, repo, task_index=0, push_adapter=True):
        model, adapter = self.model.factorize(task_index=task_index)
        model.push_to_hub(repo)
        self.tokenizer.push_to_hub(repo)
        if push_adapter:
            adapter.push_to_hub(f"{repo}-adapter")    

    def preprocess_tasks(self, tasks, tokenizer):      
        features_dict = {}
        for i, task in enumerate(tasks):
            print("[*] Preprocessing task",i,"=>",task.name)
            with NoTqdm():
                if hasattr(task, 'processed_features') and tokenizer==task.tokenizer:
                    features_dict[task]=task.processed_features
                    continue
                task.set_tokenizer(tokenizer)

                # rename the 'target' column to 'labels'
                if hasattr(task, "y") and task.y != "labels":
                    task.dataset = task.dataset.rename_column(task.y, "labels")
                
                for split in task.dataset:
                    tdp=task.dataset[split]
                    if 'task' in tdp.features:
                        tdp=tdp.remove_crteolumns('task')
                    task.index = task.dataset[split].index = i


                features_dict[task] = {}
                for phase, phase_dataset in task.dataset.items():
                    phase_dataset.index = i

                    features_dict[task][phase] = phase_dataset.map(
                        task.preprocess_function, 
                        batched = self.batched, 
                        load_from_cache_file = True,
                        num_proc = self.num_proc
                    )

                    features_dict[task][phase].set_format(
                        type="torch", columns=["input_ids", "attention_mask", "labels", "task"]
                    )
                
                task.processed_features = features_dict[task] # cache the processed features
        
        return features_dict

def Model_Trainer(tasks, args):
    model = Model(tasks, args)
    trainer = Trainer(model, tasks, args)
    return model, trainer
=======
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
            labels_name = f"target_{i+1}"
            labels_i = torch.tensor([i[labels_name] for i in inputs], device=self.args.device)
            logits, loss = head(sequence_output, pooled_output, labels=labels_i, attention_mask=attention_mask)
            loss_list.append(loss)
            logits_list[labels_name] = logits
        
        loss = torch.stack(loss_list)
        return (loss, logits_list) if return_outputs else loss
>>>>>>> Stashed changes
