import numpy as np
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
import evaluate
import funcy as fc
import warnings
from frozendict import frozendict as fdict
from dataclasses import dataclass

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

    @staticmethod
    def _align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:
                new_labels.append(-100)

            elif word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            
            else:
                label = labels[word_id]
                new_labels.append(label)
        
        return new_labels

    def __post_init__(self):
        print("[*] Initializing TokenClassificationTask... with y:", self.y)
        self.label_names = {}
        self.num_labels  = {}
        
        if type(self.y) == str:
            self.y = [self.y]

        for y in self.y:
            target = self.dataset[self.main_split].features[y]
            self.num_labels[y] = target.feature.num_classes
            self.label_names[y] = target.feature.names if target.feature.names else [None]
        
        print(f"[*] TokenClassificationTask loaded {self.task_type} task with {self.num_labels} labels")
        for k,v in self.label_names.items():
            print(f"      {k} labels: {v}")

    def get_labels(self):
        return super().get_labels() or self.label_names

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.add_prefix_space = True
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer = self.tokenizer
        )

    def preprocess_function(self, examples):
        if examples[self.tokens] and type(examples[self.tokens][0]) == str:
            unsqueeze, examples = True, {k:[v] for k,v in examples.items()}
        
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