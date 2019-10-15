import os
import pickle
import argparse
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from mspan_roberta_gcn.drop_roberta_dataset import DropReader

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--passage_length_limit", type=int, default=463)
parser.add_argument("--question_length_limit", type=int, default=46)

args = parser.parse_args()

tokenizer = RobertaTokenizer.from_pretrained(args.input_path + "/roberta.large")

dev_reader = DropReader(
    tokenizer, args.passage_length_limit, args.question_length_limit
)

train_reader = DropReader(
    tokenizer, args.passage_length_limit, args.question_length_limit,
    skip_when_all_empty=["passage_span", "question_span", "addition_subtraction", "counting", ]
)


data_format = "drop_dataset_{}.json"

data_mode = ["train"]
for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))
    data = train_reader._read(dpath)
    print("Save data to {}.".format(os.path.join(args.output_dir, "cached_roberta_{}.pkl".format(dm))))
    with open(os.path.join(args.output_dir, "cached_roberta_{}.pkl".format(dm)), "wb") as f:
        pickle.dump(data, f)

data_mode = ["dev"]
for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))
    data = dev_reader._read(dpath) if dm == "dev" else train_reader._read(dpath)
    print("Save data to {}.".format(os.path.join(args.output_dir, "cached_roberta_{}.pkl".format(dm))))
    with open(os.path.join(args.output_dir, "cached_roberta_{}.pkl".format(dm)), "wb") as f:
        pickle.dump(data, f)
