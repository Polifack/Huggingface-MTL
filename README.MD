# Huggingface Multi-Task Learning Model

Multi Task System with integration with Huggingface transformrers library.
- This system will consider as 'task' each training set where the input data is different and each task can have several targets. For each task and target it will train it in a multi-task setting.
- This system will deal with token classification and sequence classification tasks.
- This system will focus on NLP tasks.


Usage:
```
python main.py --config config.json
```