import re
import json
import string
import itertools
from tqdm import tqdm
import numpy as np
from word2number.w2n import word_to_num
from typing import List, Dict, Any, Tuple
from collections import defaultdict, OrderedDict


def get_number_from_word(word, improve_number_extraction=True):
    punctuation = string.punctuation.replace('-', '')
    word = word.strip(punctuation)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if improve_number_extraction:
                    if re.match('^\d*1st$', word):  # ending in '1st'
                        number = int(word[:-2])
                    elif re.match('^\d*2nd$', word):  # ending in '2nd'
                        number = int(word[:-2])
                    elif re.match('^\d*3rd$', word):  # ending in '3rd'
                        number = int(word[:-2])
                    elif re.match('^\d+th$', word):  # ending in <digits>th
                        # Many occurrences are when referring to centuries (e.g "the *19th* century")
                        number = int(word[:-2])
                    elif len(word) > 1 and word[-2] == '0' and re.match('^\d+s$', word):
                        # Decades, e.g. "1960s".
                        # Other sequences of digits ending with s (there are 39 of these in the training
                        # set), do not seem to be arithmetically related, as they are usually proper
                        # names, like model numbers.
                        number = int(word[:-1])
                    elif len(word) > 4 and re.match('^\d+(\.?\d+)?/km[²2]$', word):
                        # per square kilometer, e.g "73/km²" or "3057.4/km2"
                        if '.' in word:
                            number = float(word[:-4])
                        else:
                            number = int(word[:-4])
                    elif len(word) > 6 and re.match('^\d+(\.?\d+)?/month$', word):
                        # per month, e.g "1050.95/month"
                        if '.' in word:
                            number = float(word[:-6])
                        else:
                            number = int(word[:-6])
                    else:
                        return None
                else:
                    return None
    return number


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def roberta_tokenize(text, tokenizer, is_answer=False):
    split_tokens = []
    sub_token_offsets = []

    numbers = []
    number_indices = []
    number_len = []


    word_piece_mask = []
    # char_to_word_offset = []
    word_to_char_offset = []
    prev_is_whitespace = True
    tokens = []
    for i, c in enumerate(text):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
            word_to_char_offset.append(i)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
                word_to_char_offset.append(i)
            else:
                tokens[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)

    for i, token in enumerate(tokens):
        index = word_to_char_offset[i]
        if i != 0 or is_answer:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)
        token_number = get_number_from_word(token)

        if token_number is not None:
            numbers.append(token_number)
            number_indices.append(len(split_tokens))
            number_len.append(len(sub_tokens))

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            sub_token_offsets.append((index, index + len(token)))

        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)

    assert len(split_tokens) == len(sub_token_offsets)
    return split_tokens, sub_token_offsets, numbers, number_indices, number_len, word_piece_mask


def clipped_passage_num(number_indices, number_len, numbers_in_passage, plen):
    if len(number_indices) < 1 or number_indices[-1] < plen:
        return number_indices, number_len, numbers_in_passage
    lo = 0
    hi = len(number_indices) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if number_indices[mid] < plen:
            lo = mid + 1
        else:
            hi = mid
    if number_indices[lo - 1] + number_len[lo - 1] > plen:
        number_len[lo - 1] = plen - number_indices[lo - 1]
    return number_indices[:lo], number_len[:lo], numbers_in_passage[:lo]


def cached_path(file_path):
    return file_path

IGNORED_TOKENS = {'a', 'an', 'the'}
MULTI_SPAN = 'multi_span'
STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_"])
USTRIPPED_CHARACTERS = ''.join([u"Ġ"])
START_CHAR = u"Ġ"

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip().lower()
    if not text:
        return []
    tokens = text.split()
    tokens = [token.strip(STRIPPED_CHARACTERS) for token in tokens]
    return tokens

WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}

class DropReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None,
                 skip_when_all_empty: List[str] = None, instance_format: str = "drop",
                 relaxed_span_match_for_finding_labels: bool = True) -> None:
        self.max_pieces = 512
        self._tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_when_all_empty = skip_when_all_empty if skip_when_all_empty is not None else []
        for item in self.skip_when_all_empty:
            assert item in ["passage_span", "question_span", "addition_subtraction",
                            "counting", "multi_span"], f"Unsupported skip type: {item}"
        self.instance_format = instance_format
        self.relaxed_span_match_for_finding_labels = relaxed_span_match_for_finding_labels
        self.flexibility_threshold = 1000

    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        print("Reading the dataset")
        instances, skip_count = [], 0
        for passage_id, passage_info in tqdm(dataset.items()):
            passage_text = passage_info["passage"]
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]

                instance = self.text_to_instance(question_text, passage_text, question_id, passage_id,
                                                 answer_annotations)
                if instance is not None:
                    instances.append(instance)
                else:
                    skip_count += 1
        print(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        return instances

    def text_to_instance(self,
                         question_text: str, passage_text: str,
                         question_id: str, passage_id: str,
                         answer_annotations):
        passage_text = " ".join(whitespace_tokenize(passage_text))
        question_text = " ".join(whitespace_tokenize(question_text))

        passage_tokens, passage_offset, numbers_in_passage, number_indices, number_len, passage_wordpiece_mask =  \
            roberta_tokenize(passage_text, self._tokenizer)
        question_tokens, question_offset, numbers_in_question, question_number_indices, question_number_len, question_wordpiece_mask = \
            roberta_tokenize(question_text, self._tokenizer)
        question_tokens = question_tokens[:self.question_length_limit]
        question_number_indices, question_number_len, numbers_in_question = clipped_passage_num(
            question_number_indices, question_number_len, numbers_in_question, len(question_tokens)
        )

        qp_tokens = ["<s>"] + question_tokens + ["</s>"] + passage_tokens
        qp_wordpiece_mask = [1] + question_wordpiece_mask + [1] + passage_wordpiece_mask
        q_len = len(question_tokens)
        if len(qp_tokens) > self.max_pieces - 1:
            qp_tokens = qp_tokens[:self.max_pieces - 1]
            passage_tokens = passage_tokens[:self.max_pieces - q_len - 3]
            passage_offset = passage_offset[:self.max_pieces - q_len - 3]
            plen = len(passage_tokens)
            number_indices, number_len, numbers_in_passage = clipped_passage_num(number_indices, number_len,
                                                                                 numbers_in_passage, plen)
            qp_wordpiece_mask = qp_wordpiece_mask[:self.max_pieces - 1]
        qp_tokens += ["</s>"]
        qp_wordpiece_mask += [1]

        answer_type: str = None
        answer_texts: List[str] = []
        if answer_annotations:
            # Currently we only use the first annotated answer here, but actually this doesn't affect
            # the training, because we only have one annotation for the train set.
            answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_annotations[0])
            answer_texts = [" ".join(whitespace_tokenize(answer_text)) for answer_text in answer_texts]

        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = []
        specific_answer_type = "single_span"
        for answer_text in answer_texts:
            answer_tokens, _, _, _, _, _ = roberta_tokenize(answer_text, self._tokenizer, True)
            if answer_type in ["span", "spans"]:
                answer_texts = list(OrderedDict.fromkeys(answer_texts))
            if answer_type == "spans" and len(answer_texts) > 1:
                specific_answer_type = MULTI_SPAN
            tokenized_answer_text = " ".join(answer_tokens)
            if tokenized_answer_text not in tokenized_answer_texts:
                tokenized_answer_texts.append(tokenized_answer_text)

        if self.instance_format == "drop":
            def get_number_order(numbers):
                if len(numbers) < 1:
                    return None
                ordered_idx_list = np.argsort(np.array(numbers)).tolist()

                rank = 0
                number_rank = []
                for i, idx in enumerate(ordered_idx_list):
                    if i == 0 or numbers[ordered_idx_list[i]] != numbers[ordered_idx_list[i - 1]]:
                        rank += 1
                    number_rank.append(rank)

                ordered_idx_rank = zip(ordered_idx_list, number_rank)

                final_rank = sorted(ordered_idx_rank, key=lambda x: x[0])
                final_rank = [item[1] for item in final_rank]

                return final_rank

            all_number = numbers_in_passage + numbers_in_question
            all_number_order = get_number_order(all_number)

            if all_number_order is None:
                passage_number_order = []
                question_number_order = []
            else:
                passage_number_order = all_number_order[:len(numbers_in_passage)]
                question_number_order = all_number_order[len(numbers_in_passage):]

            number_indices = [indice + 1 for indice in number_indices]
            numbers_in_passage.append(100)
            number_indices.append(0)
            passage_number_order.append(-1)

            # hack to guarantee minimal length of padded number
            numbers_in_passage.append(0)
            number_indices.append(-1)
            passage_number_order.append(-1)

            numbers_in_question.append(0)
            question_number_indices.append(-1)
            question_number_order.append(-1)

            passage_number_order = np.array(passage_number_order)
            question_number_order = np.array(question_number_order)

            numbers_as_tokens = [str(number) for number in numbers_in_passage]

            valid_passage_spans = self.find_valid_spans(passage_tokens,
                                                        tokenized_answer_texts) if tokenized_answer_texts else []
            if len(valid_passage_spans) > 0:
                valid_question_spans = []
            else:
                valid_question_spans = self.find_valid_spans(question_tokens,
                                                         tokenized_answer_texts) if tokenized_answer_texts else []


            target_numbers = []
            # `answer_texts` is a list of valid answers.
            for answer_text in answer_texts:
                number = get_number_from_word(answer_text, True)
                if number is not None:
                    target_numbers.append(number)
            valid_signs_for_add_sub_expressions: List[List[int]] = []
            valid_counts: List[int] = []
            if answer_type in ["number", "date"]:
                target_number_strs = ["%.3f" % num for num in target_numbers]
                valid_signs_for_add_sub_expressions = self.find_valid_add_sub_expressions(numbers_in_passage,
                                                                                          target_number_strs)
            if answer_type in ["number"]:
                # Currently we only support count number 0 ~ 9
                numbers_for_count = list(range(10))
                valid_counts = self.find_valid_counts(numbers_for_count, target_numbers)

            # add multi_span answer extraction
            no_answer_bios = [0] * len(qp_tokens)
            if specific_answer_type == MULTI_SPAN and (len(valid_passage_spans) > 0 or len(valid_question_spans) > 0):
                spans_dict = {}
                text_to_disjoint_bios = []
                flexibility_count = 1
                for tokenized_answer_text in tokenized_answer_texts:
                    spans = self.find_valid_spans(qp_tokens, [tokenized_answer_text])
                    if len(spans) == 0:
                        # possible if the passage was clipped, but not for all of the answers
                        continue
                    spans_dict[tokenized_answer_text] = spans

                    disjoint_bios = []
                    for span_ind, span in enumerate(spans):
                        bios = create_bio_labels([span], len(qp_tokens))
                        disjoint_bios.append(bios)

                    text_to_disjoint_bios.append(disjoint_bios)
                    flexibility_count *= ((2 ** len(spans)) - 1)

                answer_as_text_to_disjoint_bios = text_to_disjoint_bios

                if (flexibility_count < self.flexibility_threshold):
                    # generate all non-empty span combinations per each text
                    spans_combinations_dict = {}
                    for key, spans in spans_dict.items():
                        spans_combinations_dict[key] = all_combinations = []
                        for i in range(1, len(spans) + 1):
                            all_combinations += list(itertools.combinations(spans, i))

                    # calculate product between all the combinations per each text
                    packed_gold_spans_list = itertools.product(*list(spans_combinations_dict.values()))
                    bios_list = []
                    for packed_gold_spans in packed_gold_spans_list:
                        gold_spans = [s for sublist in packed_gold_spans for s in sublist]
                        bios = create_bio_labels(gold_spans, len(qp_tokens))
                        bios_list.append(bios)

                    answer_as_list_of_bios = bios_list
                    answer_as_text_to_disjoint_bios = [[no_answer_bios]]
                else:
                    answer_as_list_of_bios = [no_answer_bios]

                # END

                # Used for both "require-all" BIO loss and flexible loss
                bio_labels = create_bio_labels(valid_question_spans + valid_passage_spans, len(qp_tokens))
                span_bio_labels = bio_labels

                is_bio_mask = 1

                multi_span = [is_bio_mask, answer_as_text_to_disjoint_bios, answer_as_list_of_bios, span_bio_labels]
            else:
                multi_span = []


            valid_passage_spans = valid_passage_spans if specific_answer_type != MULTI_SPAN or len(multi_span) < 1 else []
            valid_question_spans = valid_question_spans if specific_answer_type != MULTI_SPAN or len(multi_span) < 1 else []

            type_to_answer_map = {"passage_span": valid_passage_spans, "question_span": valid_question_spans,
                                  "addition_subtraction": valid_signs_for_add_sub_expressions, "counting": valid_counts,
                                  "multi_span": multi_span}

            if self.skip_when_all_empty and not any(
                type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty):
                # logger.info('my_skip_ans_type: %s' % answer_type)
                return None

            answer_info = {"answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                           "answer_passage_spans": valid_passage_spans,
                           "answer_question_spans": valid_question_spans,
                           "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions, "counts": valid_counts,
                           "multi_span": multi_span}

            return self.make_marginal_drop_instance(question_tokens,
                                                    passage_tokens,
                                                    qp_tokens,
                                                    numbers_as_tokens,
                                                    number_indices,
                                                    passage_number_order,
                                                    question_number_order,
                                                    question_number_indices,
                                                    qp_wordpiece_mask,
                                                    answer_info,
                                                    additional_metadata={"original_passage": passage_text,
                                                                         "passage_token_offsets": passage_offset,
                                                                         "original_question": question_text,
                                                                         "question_token_offsets": question_offset,
                                                                         "original_numbers": numbers_in_passage,
                                                                         "passage_id": passage_id,
                                                                         "question_id": question_id,
                                                                         "answer_info": answer_info,
                                                                         "answer_annotations": answer_annotations})
        else:
            raise ValueError(f"Expect the instance format to be \"drop\", \"squad\" or \"bert\", "
                             f"but got {self.instance_format}")

    @staticmethod
    def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key] for key in ["month", "day", "year"] if
                           key in answer_content and answer_content[key]]
            answer_texts = date_tokens
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def find_valid_spans(passage_tokens: List[str], answer_texts: List[str]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.strip(USTRIPPED_CHARACTERS) for token in passage_tokens]
        # normalized_tokens = passage_tokens
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = [token.strip(USTRIPPED_CHARACTERS) for token in answer_text.split()]
            # answer_tokens = answer_text.split()
            num_answer_tokens = len(answer_tokens)
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index] == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

    @staticmethod
    def find_valid_add_sub_expressions(numbers: List, targets: List, max_number_of_numbers_to_consider: int = 3) -> \
    List[List[int]]:
        valid_signs_for_add_sub_expressions = []
        # TODO: Try smaller numbers?
        for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
            possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
            for number_combination in itertools.combinations(enumerate(numbers), number_of_numbers_to_consider):
                indices = [it[0] for it in number_combination]
                values = [it[1] for it in number_combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    # if eval_value in targets:
                    eval_value_str = '%.3f' % eval_value
                    if eval_value_str in targets:
                        labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = 1 if sign == 1 else 2  # 1 for positive, 2 for negative
                        valid_signs_for_add_sub_expressions.append(labels_for_numbers)
        return valid_signs_for_add_sub_expressions

    @staticmethod
    def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices

    @staticmethod
    def make_marginal_drop_instance(question_tokens: List[str],
                                    passage_tokens: List[str],
                                    question_passage_tokens: List[str],
                                    number_tokens: List[str],
                                    number_indices: List[int],
                                    passage_number_order: np.ndarray,
                                    question_number_order: np.ndarray,
                                    question_number_indices: List[int],
                                    wordpiece_mask: List[int],
                                    answer_info: Dict[str, Any] = None,
                                    additional_metadata: Dict[str, Any] = None):
        metadata = {
                    "question_tokens": [token for token in question_tokens],
                    "passage_tokens": [token for token in passage_tokens],
                    "question_passage_tokens": question_passage_tokens,
                    "number_tokens": [token for token in number_tokens],
                    "number_indices": number_indices,
                    "question_number_indices": question_number_indices,
                    "passage_number_order": passage_number_order,
                    "question_number_order": question_number_order,
                    "wordpiece_mask": wordpiece_mask
                    }
        if answer_info:
            metadata["answer_texts"] = answer_info["answer_texts"]
            metadata["answer_passage_spans"] = answer_info["answer_passage_spans"]
            metadata["answer_question_spans"] = answer_info["answer_question_spans"]
            metadata["signs_for_add_sub_expressions"] = answer_info["signs_for_add_sub_expressions"]
            metadata["counts"] = answer_info["counts"]
            metadata["multi_span"] = answer_info["multi_span"]

        metadata.update(additional_metadata)
        return metadata


def create_bio_labels(spans: List[Tuple[int, int]], n_labels: int):

    # initialize all labels to O
    labels = [0] * n_labels

    for span in spans:
        start = span[0]
        end = span[1]
        # create B labels
        labels[start] = 1
        # create I labels
        labels[start+1:end+1] = [2] * (end - start)

    return labels
