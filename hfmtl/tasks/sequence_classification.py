import numpy as np
import datasets as ds
from datasets import load_dataset, Dataset
from transformers import DefaultDataCollator
import evaluate
import funcy as fc
import evaluate
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from scipy.special import expit
from .task import Task

def get_len(outputs):
    try:
        return len(outputs[fc.first(outputs)])
    except:
        return 1
    
@dataclass
class SequenceClassification(Task):
    task_type = "SequenceClassification"
    name: str = "SequenceClassificationTask"
    dataset: Dataset = None
    data_collator = DefaultDataCollator()
    tokens: str = 'tokens'
    y: str|list = 'target'
    num_labels: int = None
    label_names: dict = None

    def __post_init__(self):
        super().__post_init__()
        print("[*] Initializing SequenceClassificationTask... with y:", self.y)
        self.label_names = {}
        self.num_labels  = {}
        
        if type(self.y) == str:
            self.y = [self.y]

        for y in self.y:
            target = self.dataset[self.main_split].features[y]
            self.num_labels[y] = target.feature.num_classes
            self.label_names[y] = target.feature.names if target.feature.names else [None]
        
        print(f"[*] SequenceClassificationTask loaded {self.task_type} task with {self.num_labels} labels")
        for k,v in self.label_names.items():
            print(f"      {k} labels: {v}")

    @staticmethod
    def _align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:
                new_labels.append(-100)
            else:
                label = labels[word_id]
                new_labels.append(label)
        return new_labels

    def check(self):
        features = self.dataset[self.main_split].features
        return self.s1 in features and self.y in features

    def preprocess_function(self, examples):
        if examples[self.tokens] and type(examples[self.tokens][0]) == str:
            unsqueeze, examples = True, {k:[v] for k,v in examples.items()}
        tokenized_inputs = self.tokenizer(
            examples[self.tokens],
            is_split_into_words=True,
            **self.tokenizer_kwargs
        )
        tokenized_inputs = self.tokenizer(
            examples[self.tokens],
            is_split_into_words=True,
            **self.tokenizer_kwargs
        )

        for target_column in self.y:
            all_labels = examples[target_column]
            new_labels = []
            
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(self._align_labels_with_tokens(labels, word_ids))
            
            tokenized_inputs[target_column] = new_labels        
            tokenized_inputs['task_ids'] = [self.index]*get_len(tokenized_inputs)
        return tokenized_inputs

    def compute_metrics(self, eval_pred):
        avg={}
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        if "int" in str(eval_pred.label_ids.dtype):
            metric = evaluate.load("super_glue", "cb")
            predictions = np.argmax(predictions, axis=1)
            
        elif getattr(self,"problem_type", None)=='multi_label_classification':
            metric=evaluate.load('f1','multilabel', average='macro')
            labels=labels.astype(int)
            predictions = (expit(predictions)>0.5).astype(int)
            avg={"average":"macro"}
        else:
            metric = evaluate.load("glue", "stsb")
        
        meta = {"name": self.name, "size": len(predictions), "index": self.index}
        metrics = metric.compute(predictions=predictions, references=labels,**avg)
        self.results+=[metrics]
        return {**metrics, **meta}
