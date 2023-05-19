import torch
import torch.nn as nn
import argparse
import pdb
from collections import OrderedDict
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BertForSequenceClassification, Adafactor

parser = argparse.ArgumentParser()

parser.add_argument(
    "--base_trained",
    type=str,
    default="kykim/bert-kor-base",
    help="Pretrainned model from 🤗",
)
parser.add_argument("--pretrained_model", type=str, help="User trained model")


args = parser.parse_args()


def load_trained(args, model):
    state_dict = torch.load(args.pretrained_model)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    print("load safely")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained)
    model = BertForSequenceClassification.from_pretrained(
        args.base_trained, num_labels=2
    ).to("cuda")
    if args.pretrained_model:
        load_trained(args, model)
    model = nn.DataParallel(model)
    while True:
        victim_text = input("victim:")
        criminal_text = input("criminal:")
        input_text = "[피해자]" + victim_text + "[범죄자]" + criminal_text
        intent_input = "intent: [SEP]" + input_text
        belief_input = "belief: [SEP]" + input_text
        encoded_intent = tokenizer.encode(intent_input, return_tensors="pt")
        encoded_belief = tokenizer.encode(belief_input, return_tensors="pt")
        model.eval()
        label_intent = (
            model(input_ids=encoded_intent.to("cuda")).logits.argmax(axis=-1)[0].cpu()
        )
        label_belief = (
            model(input_ids=encoded_belief.to("cuda")).logits.argmax(axis=-1)[0].cpu()
        )
        print("intent label:", label_intent)
        print("belief label:", label_belief)
