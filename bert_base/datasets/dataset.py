# *_*coding:utf-8 *_*
# @Time    : 2022/7/3 22:41
# @Author  : XieSJ
# @FileName: dataset.py.py
# @Description:
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import TensorDataset
from transformers import InputFeatures


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification

    Args:
        guid: Unique id for the example
        text_a: String.The untokenized text of the first sequence.For single sequence tasks, only this sequence must be specified.
        text_b: (Optional) String. The untokenized text of the second sequence.Only must be specified for sequece pair tasks.
        label:  (Optional) String. The label of the example. This should be specified for train and dev examples,but not for test examples.
    """
    guid: int
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self), indent=2) + '\n'


class CommonProcessor():
    train_file = ""
    dev_file = ""
    test_file = ""
    label_file = ""

    def get_train_examples(self, data_dir):
        """
        Gets a collection of InputExamples for the train set.
        :param data_dir:
        :return:
        """
        train_file = os.path.join(data_dir, "train.csv")
        return self._create_samples(self._read_csv(train_file))

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of InputExamples for the train set.
        :param data_dir:
        :return:
        """
        train_file = os.path.join(data_dir, "dev.csv")
        return self._create_samples(self._read_csv(train_file))

    def get_test_examples(self, data_dir):
        """
        Gets a collection of InputExamples for the train set.
        :param data_dir:
        :return:
        """
        train_file = os.path.join(data_dir, "test.csv")
        return self._create_samples(self._read_csv(train_file))

    def _read_csv(self, orig_file):
        lines = []
        with open(orig_file, encoding="utf-8") as f:
            for line in f:
                lines.append(line.strip().split('\t'))
        return lines

    def _create_samples(self, lines):
        """
        Creates examples for the training and dev stes
        :param lines:
        :return:
        """
        examples = []
        for idx, line in enumerate(lines):
            guid = idx
            label = line[1]
            text_a = line[0]
            # text_b = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a,label=int(label)))
            # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,label=int(label)))
        return examples

    def get_labels(self, data_dir):
        """
        Get the list of labels for this data set
        :param data_dir:
        :return:
        """
        label_file = os.path.join(data_dir, 'label.txt')
        labels = []
        with open(label_file, encoding='utf-8') as f:
            for line in f:
                labels.append(line.strip())
        return labels


TASK = {'Common': CommonProcessor}


def load_and_cache_examples(args, tokenizer, evaluate=False, test=False, task="Common",examples=None,label_list=None,predictors=None,teacher_tokenizer=None):
    processor = TASK[task]()
    output_mode = "classification"
    # Load data features from cache or dataset file
    if not label_list:
        label_list = processor.get_labels(args.data_dir)
    if not examples:
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)

    batch_size_rdrop = 1 if evaluate else args.batch_size_rdrop
    features = glue_convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=tokenizer.pad_token_id,
        batch_size_rdrop=batch_size_rdrop
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([int(f.label) for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    if predictors is None:
        return dataset
    else:
        teacher_probs=None
        for predictor in predictors:
            teacher_prob=torch.tensor(predictor.teacher_probs_cal(dataset),dtype=torch.float)
            if teacher_prob is None:
                teacher_probs=teacher_prob
            else:
                teacher_probs+=teacher_prob
        teacher_probs_mean=teacher_probs/len(predictors)
        dataset=TensorDataset(all_input_ids,all_attention_mask,all_token_type_ids,all_labels,teacher_probs_mean)
    return dataset


def glue_convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_token: int = 0,
        pad_token_segment_id: int = 0,
        mask_padding_with_zero: bool = True,
        batch_size_rdrop: int = 1
):
    label_map = {label: i for label, i in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length,
                                       truncation="longest_first", return_token_type_ids=True)
        input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * padding_length
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, f'Error with input length {len(input_ids)} vs {max_length})'
        assert len(attention_mask) == max_length, f'Error with input length {len(attention_mask)} vs {max_length})'
        assert len(token_type_ids) == max_length, f'Error with input length {len(token_type_ids)} vs {max_length})'

        label = label_map[example.label]

        for _ in range(batch_size_rdrop):
            features.append(
                InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                              label=label)
            )
    return features
