import numpy as np
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
import evaluate
import funcy as fc
from dataclasses import dataclass
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
    y: str = 'labels'
    num_labels: int = None

    @staticmethod
    def _align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:
                # special token
                new_labels.append(-100)

            elif word_id != current_word:
                # new word start
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            
            
            else:
                # same word as previous token
                label = labels[word_id]

                # NER: if the label is B-XXX we change it to I-XXX
                # print(label)
                # if label % 2 == 1:
                #     label += 1
                
                new_labels.append(label)
        
        return new_labels

    def __post_init__(self):
        super().__post_init__()
        if not self.num_labels:
            # regression check
            if "float" in self.dataset[self.main_split].features[self.y].dtype:
                self.num_labels = 1
            else:
                self.num_labels = len(self.dataset[self.main_split][self.y])
        
        self.label_names = [f"{i}" for i in range(self.num_labels)]

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.add_prefix_space = True
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )

    def preprocess_function(self, examples):
        if examples[self.tokens] and type(examples[self.tokens][0]) == str:
            unsqueeze, examples = True, {k:[v] for k,v in examples.items()}
        
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
            outputs={k:v[0] for k,v in outputs.items()}
        outputs['task']=[self.index]*get_len(outputs)
        
        return outputs

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)

        true_labels = [
            [self.label_names[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric.compute(
            predictions=true_predictions, references=true_labels
        )
        meta = {"name": self.name, "size": len(predictions), "index": self.index}
        metrics = {k.replace("overall_",""):v for k,v in all_metrics.items() if "overall" in k}
        self.results+=[metrics]
        return {**metrics, **meta}

    def check(self):
        features = self.dataset['train'].features
        return self.tokens in features and self.y in features