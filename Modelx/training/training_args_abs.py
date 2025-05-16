from transformers import TrainingArguments

def get_training_args(config):
    from transformers import TrainingArguments
    import inspect

    print("TrainingArguments loaded from:", inspect.getfile(TrainingArguments))

    return TrainingArguments(
        output_dir=config.training.output_dir,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        num_train_epochs=config.training.epochs,
        eval_strategy=config.training.evaluation_strategy,  # Updated to eval_strategy
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        load_best_model_at_end=config.training.load_best_model_at_end,
        logging_steps=config.training.logging_steps,
        report_to=config.training.report_to,
        fp16=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )