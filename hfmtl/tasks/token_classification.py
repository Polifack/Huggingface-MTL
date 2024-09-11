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
    def _align_labels_with_tokens(labels, restored_tokens):
        new_labels = []
        current_word = 0

        for i in restored_tokens:
            # special tokens
            if i in ("<s>", "</s>", "<pad>", "<unk>"):
                new_labels.append(-100)

            # roberta tokenizer starts new words with 'Ġ'
            # so we need to add the corresponding label
            elif i.startswith("Ġ"):
                current_word += 1
                new_labels.append(labels[current_word])
                
            # when having a 'continuation' subword we just 
            # append the label of the previous
            else:
                new_labels.append(labels[current_word])
        
        return new_labels

    def __post_init__(self):
        print("[*] Initializing TokenClassificationTask with target:", self.y)
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
        def get_len(outputs):
            try:
                return len(outputs[fc.first(outputs)])
            except:
                return 1

        tokenized_inputs = self.tokenizer(
            examples['sentence'],
            **self.tokenizer_kwargs
        )

        # fix tokenized_inputs attention_mask for roberta

        tokenized_inputs['attention_mask'] = [
            [int(i != self.tokenizer.pad_token_id) for i in input_ids] for input_ids in tokenized_inputs['input_ids']
        ]
            
        
        for target_column in self.y:
            all_labels = examples[target_column]
            new_labels = []
            
            for i, labels in enumerate(all_labels):
                sentence_input_ids = tokenized_inputs['input_ids'][i]
                restored_tokens = self.tokenizer.convert_ids_to_tokens(sentence_input_ids)
                new_labels.append(self._align_labels_with_tokens(labels, restored_tokens))
            
            tokenized_inputs[target_column] = new_labels        
            tokenized_inputs['task_ids'] = [self.index]*get_len(tokenized_inputs)

        return tokenized_inputs
    
    def decode_labels(self, labels):
        return self.dataset[self.main_split].features[self.y[0]].feature.decode_batch(labels)