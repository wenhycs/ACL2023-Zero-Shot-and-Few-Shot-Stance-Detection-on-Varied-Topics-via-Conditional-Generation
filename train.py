import os
import torch
import random
import logging
import configargparse
import torch
import numpy as np
import subprocess
from tqdm import tqdm

from typing import List, Optional
from math import ceil
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score

from models import prepare_model
from data_utils import StanceData

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

def _get_validated_args(input_args: Optional[List[str]] = None):
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Choose a pretrained base model.")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Choose a model.")
    parser.add_argument("--model_weights", type=str, default=None,
                        help="The trained model weights.")
    parser.add_argument("--cache_dir", type=str,
                        help="Cache directory for pretrained models.")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for parameters from trained model.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for each running.")
    parser.add_argument("--update_batch_size", type=int, default=16,
                        help="Batch size for each update.")
    parser.add_argument("--lr", type=float, default=4e-5,
                        help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=30,
                        help="The number of epochs for pretrained model.")
    parser.add_argument("--num_warmup_ratio", type=float, default=0.1,
                        help="The number of steps for the warmup phase")

    parser.add_argument("--do_train", action='store_true',
                        help="Perform training")
    parser.add_argument("--predict_topic", action='store_true',
                        help="Adding generation data for predicting topics.")
    parser.add_argument("--predict_topic_neg", action='store_true',
                        help="Adding negative generation data for predicting topics.")
    parser.add_argument("--predict_stance_neg", action='store_true',
                        help="Adding negative generation data for predicting stance.")
    parser.add_argument("--remove_predict_stance", action='store_true',
                        help="Remove the data for learning predicting stance only.")

    parser.add_argument("--wiki_path", type=str, default=None,
                        help="Path to Wikipedia summaries related to the topics.")

    args = parser.parse_args(input_args)
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    
    return args


def setup_cuda_device(no_cuda=False):
    if no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    else:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    return device, n_gpu


def set_random_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Set random seed to {seed}")


def setup_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    logging.info("***** Setting up tokenizer *****\n")

    model_path = model_name
    if cache_dir:
        logging.info("Loading tokenizer from cache files")
        model_path = os.path.join(cache_dir, model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"No cache directory {model_path} exists.")
    else:
        logging.info(f"Loading {model_path} tokenizer")
        print(cache_dir)
    do_lower_case = 'uncased' in model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, do_lower_case=do_lower_case)

    return tokenizer


def setup_scheduler_optimizer(model: torch.nn.Module, num_warmup_ratio: float, num_training_steps: int,
                              lr: Optional[float] = 2e-5,
                              weight_decay: Optional[float] = 0):
    logging.info("***** Setting up scheduler and optimizer *****\n")

    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in parameters_to_optimize if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in parameters_to_optimize if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    num_warmup_steps = ceil(num_warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=num_training_steps)
    
    logging.info(f"Initialized optimizer {optimizer}")
    logging.info(f"Initialized scheduler {scheduler}")

    return scheduler, optimizer


def map_gen_to_prediction(decoded_text):
    decoded_text = decoded_text.split(" ")
    # Default is neutral
    stance = 2
    for i in range(len(decoded_text)-2):
        if decoded_text[i] == 'Stance' and decoded_text[i + 1] == 'is':
            stance_text = decoded_text[i+2].strip(".")
            stance = 0 if stance_text == 'opposite' else 1 if stance_text == 'supportive' else 2
    return stance


def evaluate(model, dataloader, device, args):
    model.eval()
    all_logits, all_labels, all_data_type_marks = [], [], []
    predicted_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            kwargs = dict()

            if args.model_name.startswith("bart-"):
                kwargs["input_ids"] = batch["input_ids"].to(device)
                kwargs["attention_mask"] = batch["attention_mask"].to(device)
                kwargs["num_beams"] = 1
                kwargs["do_sample"] = False
                kwargs["max_length"] = 50
                labels = batch["output_input_ids"].to(device)

            if args.model_name.startswith("bart-"):
                generated_tokens = model.generate(**kwargs)
                decoded_preds = args.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = args.tokenizer.batch_decode(labels, skip_special_tokens=True)
                pred_labels = [map_gen_to_prediction(x) for x in decoded_preds]
                true_labels = [map_gen_to_prediction(x) for x in decoded_labels]
                all_labels += true_labels
                predicted_labels += pred_labels
                all_data_type_marks.append(batch["data_type_mark"])

    if args.model_name.startswith("bart-"):
        predicted_labels = np.array(predicted_labels)
        all_labels = np.array(all_labels)
        all_data_type_marks = torch.cat(all_data_type_marks, dim=0).cpu().numpy()

    metrics = dict()
    metrics["f1"] = f1_score(all_labels, predicted_labels, average='macro')
    metrics["precision"] = precision_score(all_labels, predicted_labels, average='macro')
    metrics["recall"] = recall_score(all_labels, predicted_labels, average='macro')

    for mark in [0, 1]:
        subset_predicted_labels = []
        subset_true_labels = []
        for i in range(len(all_labels)):
            if all_data_type_marks[i] == mark:
                subset_predicted_labels.append(predicted_labels[i])
                subset_true_labels.append(all_labels[i])
        metrics["f1_"+str(mark)] = f1_score(subset_true_labels, subset_predicted_labels, average='macro')
        metrics["precision_"+str(mark)] = precision_score(subset_true_labels, subset_predicted_labels, average='macro')
        metrics["recall"+str(mark)] = recall_score(subset_true_labels, subset_predicted_labels, average='macro')

    return metrics


def train(args, model, train_dataloader, dev_dataloader, test_dataloader, device, n_gpu, neg_train_dataloader):
    def process_train_batch(args, batch, device, is_neg=False):
        kwargs = dict()

        if args.model_name.startswith("bart-"):
            kwargs["input_ids"] = batch["input_ids"].to(device)
            kwargs["attention_mask"] = batch["attention_mask"].to(device)
            labels = batch["output_input_ids"].to(device)
            # Padding tokens are not involved in training
            labels[labels==args.pad_token_id] = -100
            kwargs["labels"] = labels
            if is_neg:
                kwargs["is_neg"] = True
                kwargs["neg_loss_mask"] = batch["negative_loss_mask"].to(device)

        return kwargs

    num_training_steps_per_epoch = ceil(len(train_dataloader.dataset)/float(args.update_batch_size))
    num_training_steps = args.num_train_epochs * num_training_steps_per_epoch

    scheduler, optimizer = setup_scheduler_optimizer(model=model,
                                                     num_warmup_ratio=args.num_warmup_ratio,
                                                     num_training_steps=num_training_steps,
                                                     lr=args.lr)
    if neg_train_dataloader is not None:
        neg_train_iter = iter(neg_train_dataloader)
    best_f1 = 0.
    update_per_batch = int(args.update_batch_size / args.batch_size)
    for epoch in range(1, args.num_train_epochs+1, 1):
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader),
                             desc=f'Running train for epoch {epoch}',
                             total=len(train_dataloader)):
            kwargs = process_train_batch(args, batch, device)
            outputs = model(**kwargs)
            loss = outputs.loss
            loss /= update_per_batch
            loss.backward()

            if neg_train_dataloader is not None:
                try:
                    neg_batch = next(neg_train_iter)
                except StopIteration:
                    neg_train_iter = iter(neg_train_dataloader)
                    neg_batch = next(neg_train_iter)
                neg_kwargs = process_train_batch(args, neg_batch, device, is_neg=True)
                neg_outputs = model(**neg_kwargs)
                loss = neg_outputs.loss
                loss = loss / (2 * update_per_batch)
                loss.backward()

            if (i+1) % update_per_batch == 0 or (i+1) == len(train_dataloader):
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        logging.info(f"Evaluation for epoch {epoch}")
        dev_metrics = evaluate(model, dev_dataloader, device, args)
        log_text = ""
        for k in dev_metrics:
            log_text += k + ":" + str(dev_metrics[k]) + ","
        logging.info(log_text)

        if dev_metrics["f1"] > best_f1:
            logging.info(f"New best, dev_f1={dev_metrics['f1']} > best_f1={best_f1}")
            best_f1 = dev_metrics["f1"]
            model.save_pretrained(args.output_dir)

def main(input_args: Optional[List[str]] = None):
    args = _get_validated_args(input_args)
    try:
        logging.info(f"Current Git Hash: {get_git_revision_hash()}")
    except:
        pass

    device, n_gpu = setup_cuda_device()
    set_random_seed(args.seed, n_gpu)
    model_name = args.model_name if args.cache_dir is None else \
                 os.path.join(args.cache_dir, args.model_name)
    tokenizer = setup_tokenizer(model_name=args.model_name, cache_dir=args.cache_dir)
    args.pad_token_id = tokenizer.pad_token_id
    args.tokenizer = tokenizer

    kwargs = dict()
    kwargs["wiki_path"] = args.wiki_path
    train_data = StanceData(data_name="./data/VAST/vast_train.csv", tokenizer=tokenizer, **kwargs)
    dev_data = StanceData(data_name="./data/VAST/vast_dev.csv", tokenizer=tokenizer, **kwargs)
    test_data = StanceData(data_name="./data/VAST/vast_test.csv", tokenizer=tokenizer, **kwargs)

    if not args.remove_predict_stance:
        train_data_list = [train_data]
    else:
        train_data_list = []
    if args.predict_topic:
        kwargs["data_augment"] = "predict_topic"
        topic_train_data = StanceData(data_name="./data/VAST/vast_train.csv", tokenizer=tokenizer, **kwargs)
        train_data_list.append(topic_train_data)
    neg_train_data_list = list()
    if args.predict_stance_neg:
        kwargs["data_augment"] = "predict_stance_neg_1"
        topic_train_neg_data = StanceData(data_name="./data/VAST/vast_train.csv", tokenizer=tokenizer, **kwargs)
        neg_train_data_list.append(topic_train_neg_data)
        kwargs["data_augment"] = "predict_stance_neg_2"
        topic_train_neg_data = StanceData(data_name="./data/VAST/vast_train.csv", tokenizer=tokenizer, **kwargs)
        neg_train_data_list.append(topic_train_neg_data)
    if args.predict_topic_neg:
        kwargs["data_augment"] = "predict_topic_neg_1"
        topic_train_neg_data = StanceData(data_name="./data/VAST/vast_train.csv", tokenizer=tokenizer, **kwargs)
        neg_train_data_list.append(topic_train_neg_data)
        kwargs["data_augment"] = "predict_topic_neg_2"
        topic_train_neg_data = StanceData(data_name="./data/VAST/vast_train.csv", tokenizer=tokenizer, **kwargs)
        neg_train_data_list.append(topic_train_neg_data)
    if len(neg_train_data_list) == 0:
        neg_train_dataloader = None
    else:
        neg_train_data = torch.utils.data.ConcatDataset(neg_train_data_list)
        neg_train_dataloader = DataLoader(neg_train_data, batch_size=args.batch_size, shuffle=True)

    train_data = torch.utils.data.ConcatDataset(train_data_list)

    model = prepare_model(model_name, args.model_type, device, args.model_weights, tokenizer)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if args.do_train:
        train(args, model, train_dataloader, dev_dataloader, test_dataloader, device, n_gpu, neg_train_dataloader)

    # Load the best model
    model = prepare_model(model_name, args.model_type, device, args.output_dir, tokenizer)
    # Evaluate on the test set
    test_metrics = evaluate(model, test_dataloader, device, args)
    log_text = "Test results: "
    for k in test_metrics:
        log_text += k + ":" + str(test_metrics[k]) + ","
    logging.info(log_text)



if __name__ == "__main__":
    main()
