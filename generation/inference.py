import argparse
from inference_pipeline import InferencePipeline

def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/ClassEval_data.json",
        help="ClassEval data",
    )
    parser.add_argument(
        "--greedy",
        type=int,
        default=1,
        help="Whether to generate model results with greedy strategy",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="model_output.json",
        help="output file path",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        nargs="+",  # Accept one or more integers
        default=None,
        help="List of CUDA device(s), default value is None. If not set, use all available devices.",
    )
    parser.add_argument(
        "--generation_strategy",
        type=int,
        default=0,
        help="Holistic = 0, Incremental = 1, Compositional = 2",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=2,
        help="DEEPSEEK_API = 0, QWEN_CODER = 1, QWEN_CODER_INST = 2",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct",
        help="model path",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature value in generation config",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="max tokens of model's generation result",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="The number of code samples that are randomly generated for each task.",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        default="/data0/xjh/ClassEval/expriment_outputs/qwen_inst_predictions_hoslitic",
        help="pred output path",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = args_init()
    print("args:", args)
    infer = InferencePipeline(args)
    infer.pipeline()