import json
import torch
import options
import argparse
from tqdm import tqdm
from mspan_roberta_gcn.inference_batch_gen import DropBatchGen
from mspan_roberta_gcn.mspan_roberta_gcn import NumericallyAugmentedBertNet
from mspan_roberta_gcn.drop_roberta_dataset import DropReader
from tag_mspan_robert_gcn.drop_roberta_mspan_dataset import DropReader as TDropReader
from tag_mspan_robert_gcn.inference_batch_gen import DropBatchGen as TDropBatchGen
from tag_mspan_robert_gcn.tag_mspan_roberta_gcn import NumericallyAugmentedBertNet as TNumericallyAugmentedBertNet
from pytorch_transformers import RobertaTokenizer, RobertaModel, RobertaConfig


parser = argparse.ArgumentParser("Bert inference task.")
options.add_bert_args(parser)
options.add_model_args(parser)
options.add_inference_args(parser)

args = parser.parse_args()

args.cuda = torch.cuda.device_count() > 0


print("Build bert model.")
bert_model = RobertaModel(RobertaConfig().from_pretrained(args.roberta_model))
print("Build Drop model.")
if args.tag_mspan:
    network = TNumericallyAugmentedBertNet(bert_model,
                                          hidden_size=bert_model.config.hidden_size,
                                          dropout_prob=0.0,
                                          use_gcn=args.use_gcn,
                                          gcn_steps=args.gcn_steps)
else:
    network = NumericallyAugmentedBertNet(bert_model,
                hidden_size=bert_model.config.hidden_size,
                dropout_prob=0.0,
                use_gcn=args.use_gcn,
                gcn_steps=args.gcn_steps)

if args.cuda: network.cuda()
print("Load from pre path {}.".format(args.pre_path))
network.load_state_dict(torch.load(args.pre_path))

print("Load data from {}.".format(args.inf_path))
tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)
if args.tag_mspan:
    inf_iter = TDropBatchGen(args, tokenizer,
                            TDropReader(tokenizer, passage_length_limit=463, question_length_limit=46)._read(
                                args.inf_path))
else:
    inf_iter = DropBatchGen(args, tokenizer, DropReader(tokenizer, passage_length_limit=463, question_length_limit=46)._read(args.inf_path))

print("Start inference...")
result = {}
network.eval()
with torch.no_grad():
    for batch in tqdm(inf_iter):
        output_dict = network(**batch)
        for i in range(len(output_dict["question_id"])):
            result[output_dict["question_id"][i]] =  output_dict["answer"][i]["predicted_answer"]

with open(args.dump_path, "w", encoding="utf8") as f:
    json.dump(result, f)
