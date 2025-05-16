import os
import shutil

def save_best_model(trainer, output_dir):
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    trainer.save_model(best_model_dir)
    if hasattr(trainer, 'tokenizer'):
        trainer.tokenizer.save_pretrained(best_model_dir)

def clean_checkpoints(output_dir, keep_last_n=3):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]), reverse=True)
    for checkpoint in checkpoints[keep_last_n:]:
        shutil.rmtree(os.path.join(output_dir, checkpoint))