# *_*coding:utf-8 *_*
# @Time    : 2022/9/16 21:48
# @Author  : XieSJ
# @FileName: medr_config.py.py
# @Description:
import argparse


def get_tiktok_config():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="resource\\tiktok_demo",
        type=str,
        help="The input data dir.Should contrain the .tsv files(or other data files_for the task."
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        # required=True,
        help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="pre_model\\bert-base-uncased",
        type=str,
        # required=True,
        help=""
    )
    parser.add_argument(
        "--output_dir",
        default="output\\tiktok",
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--batch_size_rdrop",
        default=1,
        type=int,
        help="rdrop number"
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,sequences shorter will be padded."
    )
    """
    action关键字默认状态有两种，store_true和store_false，若输入命令时，不指定其参数，则store_true显示为False,store_false显示为True。
    若命令行输入时不指定,那么结果默认为False
    """
    parser.add_argument(
        "--do_train",
        action="store_false",
        help="Whether to run training"
    )
    parser.add_argument(
        "--do_eval",
        action="store_false",
        help="Whether to run eval on the dev set."
    )

    # for do test
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to run eval on the test set."
    )
    parser.add_argument(
        "--test_file",
        default="",
        type=str
    )
    parser.add_argument(
        "--test_output_dir",
        default="",
        type=str
    )
    parser.add_argument(
        "--save_test_logits",
        default="",
        type=str
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--learning_rate",
        default=8e-6,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some"
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=4.0,
        type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_step",
        default=-1,
        type=int,
        help="if >0:set total number of training steps to perform.Override num_train_epochs"
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps"
    )
    parser.add_argument(
        "--keep_checkpoint_max",
        type=int,
        default=3
    )
    parser.add_argument(
        "--logging_steps",
        default=200,
        type=int,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        default=200,
        type=int,
        help="Save checkpoint every X updates steps"
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_false",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Save checkpoint every X updates steps"
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed for initialization"
    )
    args=parser.parse_args()
    return args