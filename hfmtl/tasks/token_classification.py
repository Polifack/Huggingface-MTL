import numpy as np
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
<<<<<<< Updated upstream
from transformers import logging
=======
>>>>>>> Stashed changes
import evaluate
import funcy as fc
import warnings
from frozendict import frozendict as fdict
from dataclasses import dataclass
<<<<<<< Updated upstream
from .task import Task

def get_len(outputs):
    try:
        return len(outputs[fc.first(outputs)])
    except:
        return 1

@dataclass
class TokenClassification(Task):
    task_type = "TokenClassification"
    dataset: Dataset = None
    metric:... = evaluate.load("seqeval")
    tokens: str = 'tokens'
    y:      str = 'target'
    num_labels: int = None
    tokenizer_kwargs: fdict = fdict(padding="max_length", max_length=256, truncation=True)
=======

@dataclass
class TokenClassification:
    task_type = "TokenClassification"
    name: str = "TokenClassificationTask"
    dataset: Dataset = None
    metric:... = evaluate.load("seqeval")
    main_split: str = "train"
    tokens: str = 'tokens'
    y: str|list = 'target'
    num_labels: int = None
    label_names: dict = None
    tokenizer_kwargs: fdict = fdict(padding="max_length", max_length=128, truncation=True)
>>>>>>> Stashed changes

    @staticmethod
    def _align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:
<<<<<<< Updated upstream
                # special token
                new_labels.append(-100)

            elif word_id != current_word:
                # new word start
=======
                new_labels.append(-100)

            elif word_id != current_word:
>>>>>>> Stashed changes
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            
<<<<<<< Updated upstream
            
            else:
                # same word as previous token
                label = labels[word_id]

                # NER: if the label is B-XXX we change it to I-XXX
                # print(label)
                # if label % 2 == 1:
                #     label += 1
                
=======
            else:
                label = labels[word_id]
>>>>>>> Stashed changes
                new_labels.append(label)
        
        return new_labels

    def __post_init__(self):
<<<<<<< Updated upstream
        warnings.filterwarnings("ignore")
        # logging.set_verbosity_error()
        super().__post_init__()
        
        target = self.dataset[self.main_split].features[self.y]
        if not self.num_labels:
            self.num_labels = 1 if "float" in target.dtype else target.feature.num_classes
        
        self.label_names = target.feature.names
        print("[*] Number of labels found:",len(self.label_names))
=======
        self.label_names = {}
        self.num_labels  = {}

        for y in self.y:
            target = self.dataset[self.main_split].features[y]
            self.num_labels[y] = target.feature.num_classes
            self.label_names[y] = target.feature.names if target.feature.names else [None]
        
        print(f"[*] TokenClassificationTask loaded {self.task_type} task with {self.num_labels} labels")
        for k,v in self.label_names.items():
            print(f"      {k} labels: {v}")

    def get_labels(self):
        return super().get_labels() or self.label_names
>>>>>>> Stashed changes

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.add_prefix_space = True
        self.data_collator = DataCollatorForTokenClassification(
<<<<<<< Updated upstream
            tokenizer=self.tokenizer
=======
            tokenizer = self.tokenizer
>>>>>>> Stashed changes
        )

    def preprocess_function(self, examples):
        if examples[self.tokens] and type(examples[self.tokens][0]) == str:
            unsqueeze, examples = True, {k:[v] for k,v in examples.items()}
        
<<<<<<< Updated upstream
        tokenized_inputs = self.tokenizer(
            examples[self.tokens], 
            is_split_into_words=True, 
            **self.tokenizer_kwargs
        )

        all_labels = examples["labels"]
        new_labels = []
        
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))
        
        tokenized_inputs["labels"] = new_labels
        outputs = tokenized_inputs
        
        if 'unsqueeze' in locals() and unsqueeze:
            outputs = {k:v[0] for k,v in outputs.items()}
        outputs['task']=[self.index]*get_len(outputs)
        
        # print a sample
        return outputs

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        print("logits&&labels = ",len(logits), len(labels))
        predictions = np.argmax(logits, axis=-1)

        true_labels = [
            [self.label_names[l] for l in label if l != -100] for label in labels
        ]

        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        print("Computing metrics...")
        print("*** example:")
        print(len(true_labels), len(true_predictions))
        print("labels:",true_labels[0:5])
        print("predictions:", true_predictions[0:5])
        print("***")
        all_metrics = self.metric.compute(
            predictions = true_predictions, references = true_labels
        )
        meta = {"name": self.name, "size": len(predictions), "index": self.index}
        metrics = {k.replace("overall_",""):v for k,v in all_metrics.items() if "overall" in k}
        self.results+=[metrics]
        return {**metrics, **meta}

    def check(self):
        features = self.dataset['train'].features
        return self.tokens in features and self.y in features
=======
        def get_len(outputs):
            try:
                return len(outputs[fc.first(outputs)])
            except:
                return 1
        
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
>>>>>>> Stashed changes