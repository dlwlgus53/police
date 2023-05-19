import re
import pdb
import json
import torch
import pickle
import ontology
from tqdm import tqdm
import logging
from log_conf import init_logger
from collections import defaultdict
import random

logger = logging.getLogger("my")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, data_type):
        random.seed(args.seed)
        self.data_type = data_type
        self.tokenizer = args.tokenizer
        if args.do_short:
            data_path = "../POLICE/short_data.json"
        logger.info(f"load {self.data_type} data file {data_path}")
        raw_dataset = json.load(open(data_path, "r"))

        (
            self.turn_id,
            self.dial_id,
            self.input_text,
            self.question,
            self.answer,
        ) = self.seperate_data(raw_dataset)

        assert len(self.turn_id) == len(self.dial_id)

    def __len__(self):
        return len(self.dial_id)

    def seperate_data(self, dataset):
        turn_id, dial_id, input_text, question, answer = [], [], [], [], []

        for dialogue in dataset:
            d_id = dialogue["ID"]
            dialogue_text = ""
            for t_id, turn in enumerate(dialogue["log"]):
                dialogue_text += "[범죄자] "
                dialogue_text += turn["criminal"]

                for q in ["belief", "intent"]:
                    a = 0

                    if q == "belief" and len(turn["belief"]) != 0:
                        a = 1
                    if q == "intent" and 1 in turn["intent"].values():
                        a = 1

                    turn_id.append(t_id)
                    dial_id.append(d_id)
                    input_text.append(dialogue_text)
                    question.append(q)
                    answer.append(a)
                    if a == 1:
                        for _ in range(10):
                            turn_id.append(t_id)
                            dial_id.append(d_id)
                            input_text.append(dialogue_text)
                            question.append(q)
                            answer.append(a)

                # reset the history. use just current turn
                dialogue_text = "[피해자] "
                dialogue_text += turn["victim"]

        return turn_id, dial_id, input_text, question, answer

    def __getitem__(self, index):
        self.turn_id, self.dial_id, self.input_text, self.question, self.answer

        dial_id = self.dial_id[index]
        turn_id = self.turn_id[index]
        input_text = self.input_text[index]
        question = self.question[index]
        answer = self.answer[index]

        return {
            "dial_id": dial_id,
            "turn_id": turn_id,
            "input_text": input_text,
            "question": question,
            "answer": answer,
        }

    def collate_fn(self, batch):
        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        input_text = [x["input_text"] for x in batch]
        answer = torch.tensor([x["answer"] for x in batch])

        concat_text = [q + "[SEP]" + i for (q, i) in zip(question, input_text)]
        concat_text = self.tokenizer(
            concat_text, truncation=True, padding=True, return_tensors="pt"
        )

        # encode input_text with question and answer

        return {
            "input": concat_text,
            "target": answer,
            "dial_id": dial_id,
            "turn_id": turn_id,
        }


if __name__ == "__main__":
    import argparse

    init_logger(f"data_process.log")
    logger = logging.getLogger("my")

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_rate", type=float, default=1.0)
    parser.add_argument("--do_short", type=int, default=1)
    parser.add_argument("--dst_student_rate", type=float, default=0.0)
    parser.add_argument("--seed", type=float, default=1)
    parser.add_argument("--aux", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=128)

    args = parser.parse_args()

    args.data_path = "../POLICE2/train_data.json"
    from transformers import T5Tokenizer

    args.tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

    dataset = Dataset(args, args.data_path, "train")
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn
    )
    t = args.tokenizer
    for batch in loader:
        for i in range(16):
            print(t.decode(batch["input"]["input_ids"][i]))
            print(t.decode(batch["target"]["input_ids"][i]))
            print()

        pdb.set_trace()
