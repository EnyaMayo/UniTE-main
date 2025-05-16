from tensorboardX import SummaryWriter
import logging

def setup_logging(log_dir='output/runs'):
    # Console logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # TensorBoard logging
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def log_metrics(writer, metrics, step, prefix='train'):
    for key, value in metrics.items():
        writer.add_scalar(f'{prefix}/{key}', value, step)