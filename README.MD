# Huggingface Multi-Task Learning Model

Development of a MTL system for Huggingface Transformers library.

This system will consider as 'task' each training set where the input data is different and each task can have several targets. For each task and target it will train it in a multi-task setting.

Usage:
```
python main.py --config config.json
```