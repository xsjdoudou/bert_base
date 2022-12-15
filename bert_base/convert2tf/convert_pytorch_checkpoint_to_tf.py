# *_*coding:utf-8 *_*
# @Time    : 2022/12/11 0:05
# @Author  : XieSJ
# @FileName: convert_pytorch_checkpoint_to_tf.py.py
# @Description:

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint"""

from convert2tf.tf_modeling_bert import BertConfig, BertModel


class TaskType:
    masked_language_model="MaskedLM"
    sequence_labeling='SequenceLabeling' # Sequence labeling task eg.NER
    classification='Classification' # Classification task eg.sentiment classification
    machine_reading_comprehension='MachineReadingComprehension' # SQuAD式阅读理解
    multi_choice='MultiChoice' #多项式阅读理解
    con_cat="ConCat" # 拼接

    MODEL_CLASSES={
        "bert":(BertConfig,BertModel),
        "albert":(AlbertConfig)
    }

