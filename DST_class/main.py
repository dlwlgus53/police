import utils
import time
import torch
import torch.nn as nn

import logging
import argparse
import datetime
from dataset import Dataset
import init

from collections import OrderedDict
from trainer import valid, train, test
from torch.utils.data import DataLoader


from transformers import AutoTokenizer, BertForSequenceClassification, Adafactor

import pdb

parser = argparse.ArgumentParser()

"""training"""
parser.add_argument("--do_train", type=int, default=1, help="If 1, then train")
parser.add_argument("--do_short", type=int, default=1, help="If 1, then use small data")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
parser.add_argument("--test_batch_size", type=int, default=16, help="Test batch size")
parser.add_argument("--max_epoch", type=int, default=1, help="Max epoch number")
parser.add_argument(
    "--base_trained",
    type=str,
    default="kykim/bert-kor-base",
    help="Pretrainned model from 🤗",
)
parser.add_argument("--pretrained_model", type=str, help="User trained model")

"""enviroment"""
parser.add_argument("-g", "--gpus", default=4, type=int, help="number of gpus per node")

parser.add_argument("--seed", type=int, default=1, help="Training seed")

"""saving"""
parser.add_argument(
    "--save_prefix", type=str, help="prefix for all savings", default=""
)

"""data"""
parser.add_argument("--dev_path", type=str, default="../POLICE_data/dev_data.json")
parser.add_argument("--train_path", type=str, default="../POLICE_data/POLICE/train_data.json")
parser.add_argument("--test_path", type=str, default="../POLICE_data/POLICE/dev_data.json")


args = parser.parse_args()
init.init_experiment(args)
logger = logging.getLogger("my")


def load_trained(args, model, optimizer=None):
    logger.info(f"User pretrained model{args.pretrained_model}")
    state_dict = torch.load(args.pretrained_model)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    if optimizer:
        opt_path = "./model/optimizer/" + args.pretrained_model[7:]  # todo
        optimizer.load_state_dict(torch.load(opt_path))
    print("load safely")


def get_loader(dataset, batch_size):
    shuffle = False
    pin_memory = True
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=0,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
    return loader


def main():
    logger.info(args)
    args.tokenizer = AutoTokenizer.from_pretrained(args.base_trained)
    model = BertForSequenceClassification.from_pretrained(
        args.base_trained, num_labels=2
    ).to("cuda")
    if args.pretrained_model:
        load_trained(args, model)
    model = nn.DataParallel(model)

    train_dataset = Dataset(args, args.train_path, "train")
    val_dataset = Dataset(args, args.dev_path, "val")
    batch_size = int(args.batch_size / args.gpus)
    train_loader = get_loader(train_dataset, batch_size)
    dev_loader = get_loader(val_dataset, batch_size)

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    min_loss = float("inf")
    best_performance = {}

    logger.info("Trainning start")
    for epoch in range(args.max_epoch):
        logger.info(f"epoch {epoch}")
        train(args, model, train_loader, optimizer)
        loss = valid(args, model, dev_loader)
        logger.info("Epoch : %d,  Loss : %.04f" % (epoch, loss))

        if loss < min_loss:
            logger.info("New best")
            min_loss = loss
            best_performance["min_loss"] = min_loss.item()
            torch.save(model.state_dict(), f"model/police_{args.save_prefix}.pt")
            logger.info("safely saved")


if __name__ == "__main__":
    utils.makedirs("./data")
    utils.makedirs("./logs")
    utils.makedirs("./model/optimizer")
    utils.makedirs("./out")
    logger.info(f"{'-' * 30}")
    logger.info("Start New Trainning")
    start = time.time()
    main()
    result_list = str(datetime.timedelta(seconds=time.time() - start)).split(".")
    logger.info(f"take time : {result_list[0]}")
    logger.info("End The Trainning")
    logger.info(f"{'-' * 30}")
