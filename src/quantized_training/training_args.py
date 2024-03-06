from dataclasses import dataclass, field
from typing import Optional, List

from .utils import SLURM_ARGS

__all__ = [
    "QuantizedTrainingArguments",
    "add_training_args",
]

@dataclass
class QuantizedTrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or train from scratch.
    """
    num_hidden_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of encoder layers to use."}
    )
    # bf16: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to use bf16 (mixed) precision instead of 32-bit float."}
    # )
    # do_train: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to run training"}
    # )
    gpu: Optional[int] = field(
        default=None,
        metadata={"help": "GPU to use."}
    )
    sgd: bool = field(
        default=False,
        metadata={"help": "Whether to use SGD optimizer."}
    )
    # warmup_ratio: float = field(
    #     default=0.06,
    #     metadata={"help": "Ratio of warmup steps in the lr scheduler."}
    # )
    lora_rank: int = field(
        default=0,
        metadata={"help": "The dimension of the low-rank matrices."}
    )
    lora_alpha: int = field(
        default=8,
        metadata={"help": "The scaling factor for the low-rank matrices."}
    )
    target_modules: List[str] = field(
        default="query,value",
        metadata={
            "type": lambda x: x.split(','),
            "help": "The modules (for example, attention blocks) to apply the LoRA update matrices."
        },
    )
    dtype: str = field(
        default="posit8_1",
        metadata={"help": "Quantization data type to use. Choose between posit(nbits)_(es), FP8_(E4M3|E5M2), and FP8(.MIXED)."}
    )
    quantize_weights: bool = field(
        default=False,
        metadata={"help": "Whether to quantize model weights."}
    )
    quantize_fwd: Optional[str] = field(
        default=None,
        metadata={
            "nargs": "?",
            "const": "gemm",
            "help": "Whether to quantize activations. Choose from gemm, act, norm, bn, softmax, attn_scaling, and residual."
        }
    )
    quantize_bwd: Optional[str] = field(
        default=None,
        metadata={
            "nargs": "?",
            "const": "gemm",
            "help": "Whether to quantize activation gradients. Choose from gemm, act, norm, bn, softmax, attn_scaling, and residual."
        }
    )
    scaling_fwd: bool = field(
        default=False,
        metadata={"help": "Whether to quantize activation using per-tensor scaling."}
    )
    scaling_bwd: bool = field(
        default=False,
        metadata={"help": "Whether to quantize activation gradient using per-tensor scaling."}
    )
    max_fwd: float = field(
        default=64.0,
        metadata={"help": "Maximum value of a data type when performing scaling during forward pass."}
    )
    max_bwd: float = field(
        default=64.0,
        metadata={"help": "Maximum value of a data type when performing scaling during backward pass."}
    )
    amax_history_len: int = field(
        default=10,
        metadata={"help": "The length of the amax history window used for scaling factor computation."}
    )
    op_fusion: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Fuse operation with previous GEMM to reduce quantization error."}
    )
    posit_exp: bool = field(
        default=False,
        metadata={"help": "Whether to use posit approximated exponential function in softmax."}
    )
    posit_exp_shifted: bool = field(
        default=False,
        metadata={"help": "Whether to use shifted posit approximated exponential function in softmax."}
    )
    posit_reciprocal: bool = field(
        default=False,
        metadata={"help": "Whether to use posit approximated reciprocal function in softmax."}
    )
    quantize_model: bool = field(
        default=False,
        metadata={"help": "Whether to run quantized inference using defined model file."}
    )
    plot_hist: bool = field(
        default=False,
        metadata={"help": "Whether to plot the histogram of tensor value."}
    )

def add_training_args(parser):
    parser.add_argument(
        '--project',
        default=None,
        help=(
            'Optionally provide the name of the project for the project parameter '
            '(project) where you want the output of the W&B Run to be stored.'
        )
    )
    parser.add_argument(
        '--run_id',
        default=None,
        help='A unique ID for a wandb run, used for resuming.'
    )
    parser.add_argument(
        '--sweep_id',
        default=None,
        help='W&B sweep ID that includes the the entity name and the project name.'
    )
    parser.add_argument(
        '--sweep_config',
        default=None,
        help='Path to JSON file that stores sweep configuration.'
    )
    parser.add_argument("--job_count", type=int, default=None, help="Maximum number of runs to try for each batch job.")
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    parser.add_argument(
        "--log_file",
        nargs='?',
        const="",
        default=None,
        help="Set the logging file. If not specified, the log will be printed to default location."
    )
    parser.add_argument(
        "--write_script",
        choices=['bash', 'slurm'],
        default=None,
        help="Write a script for the given configuration. Choose either 'bash' or 'slurm' script.",
    )
    for k, v in SLURM_ARGS.items():
        parser.add_argument("--" + k, **v)
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="Number of encoder layers to use."
    )
    parser.add_argument("--lora_rank", type=int, default=0, help="The dimension of the low-rank matrices.")
    parser.add_argument("--lora_alpha", type=int, default=8, help="The scaling factor for the low-rank matrices.")
    parser.add_argument(
        "--target_modules",
        type=lambda x: x.split(','),
        default="query,value",
        help="The modules (for example, attention blocks) to apply the LoRA update matrices."
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 (mixed) precision instead of 32-bit float."
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--gpu", type=int, default=None, help="GPU to use.")
    parser.add_argument("--sgd", action="store_true", help="Whether to use SGD optimizer.")
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.06, help="Ratio of warmup steps in the lr scheduler."
    )
    parser.add_argument(
        "--dtype",
        default="posit8_1",
        help="Quantization data type to use. Choose between posit(nbits)_(es), FP8_(E4M3|E5M2), and FP8(.MIXED).",
    )
    parser.add_argument("--quantize_weights", action="store_true", help="Whether to quantize model weights.")
    parser.add_argument(
        "--quantize_fwd",
        nargs='?',
        const='gemm',
        default=None,
        help=(
            "Whether to quantize activations. Choose from "
            "gemm, act, norm, bn, softmax, attn_scaling, and residual."
        ),
    )
    parser.add_argument(
        "--quantize_bwd",
        nargs='?',
        const='gemm',
        default=None,
        help=(
            "Whether to quantize activation gradients. Choose from "
            "gemm, act, norm, bn, softmax, attn_scaling, and residual."
        ),
    )
    parser.add_argument(
        "--scaling_fwd",
        action="store_true",
        help="Whether to quantize activation using per-tensor scaling."
    )
    parser.add_argument(
        "--scaling_bwd",
        action="store_true",
        help="Whether to quantize activation gradient using per-tensor scaling."
    )
    parser.add_argument(
        "--max_fwd",
        type=float,
        default=64,
        help="Maximum value of a data type when performing scaling during forward pass."
    )
    parser.add_argument(
        "--max_bwd",
        type=float,
        default=64,
        help="Maximum value of a data type when performing scaling during backward pass."
    )
    parser.add_argument(
        "--amax_history_len",
        type=int,
        default=10,
        help="The length of the amax history window used for scaling factor computation."
    )
    parser.add_argument(
        "--op_fusion",
        type=lambda x: x.split(','),
        default=None,
        help="Fuse operation with previous GEMM to reduce quantization error.",
    )
    parser.add_argument(
        "--posit_exp",
        action="store_true",
        help="Whether to use posit approximated exponential function in softmax."
    )
    parser.add_argument(
        "--posit_exp_shifted",
        action="store_true",
        help="Whether to use shifted posit approximated exponential function in softmax."
    )
    parser.add_argument(
        "--posit_reciprocal",
        action="store_true",
        help="Whether to use posit approximated reciprocal function in softmax."
    )
    parser.add_argument(
        "--quantize_model",
        action="store_true",
        help="Whether to run quantized inference using defined model file.",
    )
    parser.add_argument(
        "--plot_hist",
        action="store_true",
        help="Whether to store and plot the histogram of tensor value.",
    )