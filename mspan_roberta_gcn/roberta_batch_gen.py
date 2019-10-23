import os
import pickle
import torch
import random

class DropBatchGen(object):
    def __init__(self, args, data_mode, tokenizer, padding_idx=1):
        self.args = args
        self.cls_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.padding_idx = padding_idx
        self.is_train = data_mode == "train"
        self.vocab_size = len(tokenizer)
        dpath = "cached_roberta_{}.pkl".format(data_mode)
        with open(os.path.join(args.data_dir, dpath), "rb") as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:
            question_tokens = tokenizer.convert_tokens_to_ids(item["question_tokens"])
            passage_tokens = tokenizer.convert_tokens_to_ids(item["passage_tokens"])
            all_data.append((question_tokens, passage_tokens, item))

        print("Load data size {}.".format(len(all_data)))

        self.data = DropBatchGen.make_baches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                                  self.is_train)
        self.offset = 0

    @staticmethod
    def make_baches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[:i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            q_tokens, p_tokens, metas = zip(*batch)
            bsz = len(batch)
            max_seq_len = max([len(q) + len(p) for q, p in zip(q_tokens, p_tokens)]) + 3
            max_num_len = max([1] + [len(item["number_indices"]) for item in metas])
            max_qnum_len = max([1] + [len(item["question_number_indices"]) for item in metas])

            max_pans_choice = min(8, max([1] + [len(item["answer_passage_spans"]) for item in metas]))
            max_qans_choice = min(8, max([1] + [len(item["answer_question_spans"]) for item in metas]))
            max_sign_choice = max([1] + [len(item["signs_for_add_sub_expressions"]) for item in metas])

            # qa input.
            input_ids = torch.LongTensor(bsz, max_seq_len).fill_(self.padding_idx)
            input_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
            input_segments = torch.LongTensor(bsz, max_seq_len).fill_(0)
            passage_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
            question_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)

            # number infos
            number_indices = torch.LongTensor(bsz, max_num_len).fill_(-1)
            question_number_indices = torch.LongTensor(bsz, max_qnum_len).fill_(-1)
            passage_number_order = torch.LongTensor(bsz, max_num_len).fill_(-1)
            question_number_order = torch.LongTensor(bsz, max_qnum_len).fill_(-1)

            # answer infos
            answer_as_passage_spans = torch.LongTensor(bsz, max_pans_choice, 2).fill_(-1)
            answer_as_question_spans = torch.LongTensor(bsz, max_qans_choice, 2).fill_(-1)
            answer_as_add_sub_expressions = torch.LongTensor(bsz, max_sign_choice, max_num_len).fill_(0)
            answer_as_counts = torch.LongTensor(bsz).fill_(-1)

            # answer span number
            span_num = torch.LongTensor(bsz).fill_(0)

            for i in range(bsz):
                span_num[i] = min(8, len(metas[i]["answer_texts"]))
                q_len = len(q_tokens[i])
                p_len = len(p_tokens[i])
                # input and their mask
                input_ids[i, :3 + q_len + p_len] = torch.LongTensor(
                    [self.cls_idx] + q_tokens[i] + [self.sep_idx] + p_tokens[i] + [self.sep_idx])
                input_mask[i, :3 + q_len + p_len] = 1
                question_mask[i, 1:1 + q_len] = 1
                passage_mask[i, 2 + q_len: 2 + q_len + p_len] = 1

                passage_start = q_len + 2
                question_start = 1
                # number infos
                pn_len = len(metas[i]["number_indices"]) - 1
                if pn_len > 0:
                    number_indices[i, :pn_len] = passage_start + torch.LongTensor(metas[i]["number_indices"][:pn_len])
                    passage_number_order[i, :pn_len] = torch.LongTensor(metas[i]["passage_number_order"][:pn_len])
                    number_indices[i, pn_len - 1] = 0
                qn_len = len(metas[i]["question_number_indices"]) - 1
                if qn_len > 0:
                    question_number_indices[i, :qn_len] = question_start + torch.LongTensor(
                        metas[i]["question_number_indices"][:qn_len])
                    question_number_order[i, :qn_len] = torch.LongTensor(metas[i]["question_number_order"][:qn_len])

                # answer info
                pans_len = min(len(metas[i]["answer_passage_spans"]), max_pans_choice)
                for j in range(pans_len):
                    answer_as_passage_spans[i, j, 0] = metas[i]["answer_passage_spans"][j][0] + passage_start
                    answer_as_passage_spans[i, j, 1] = metas[i]["answer_passage_spans"][j][1] + passage_start

                qans_len = min(len(metas[i]["answer_question_spans"]), max_qans_choice)
                for j in range(qans_len):
                    answer_as_question_spans[i, j, 0] = metas[i]["answer_question_spans"][j][0] + question_start
                    answer_as_question_spans[i, j, 1] = metas[i]["answer_question_spans"][j][1] + question_start

                # answer sign info
                sign_len = min(len(metas[i]["signs_for_add_sub_expressions"]), max_sign_choice)
                for j in range(sign_len):
                    answer_as_add_sub_expressions[i, j, :pn_len] = torch.LongTensor(
                        metas[i]["signs_for_add_sub_expressions"][j][:pn_len])

                # answer count info
                if len(metas[i]["counts"]) > 0:
                    answer_as_counts[i] = metas[i]["counts"][0]

            out_batch = {"input_ids": input_ids, "input_mask": input_mask, "input_segments": input_segments,
                         "passage_mask": passage_mask, "question_mask": question_mask, "number_indices": number_indices,
                         "passage_number_order": passage_number_order,
                         "question_number_order": question_number_order,
                         "question_number_indices": question_number_indices,
                         "answer_as_passage_spans": answer_as_passage_spans,
                         "answer_as_question_spans": answer_as_question_spans,
                         "answer_as_add_sub_expressions": answer_as_add_sub_expressions,
                         "answer_as_counts": answer_as_counts.unsqueeze(1), "span_num": span_num.unsqueeze(1),
                         "metadata": metas}
            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield out_batch
