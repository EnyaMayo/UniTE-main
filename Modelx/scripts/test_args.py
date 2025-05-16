from transformers import TrainingArguments
import inspect

sig = inspect.signature(TrainingArguments.__init__)
print("'evaluation_strategy' in init params:", 'evaluation_strategy' in sig.parameters)
import transformers
print("transformers module path:", transformers.__file__)
