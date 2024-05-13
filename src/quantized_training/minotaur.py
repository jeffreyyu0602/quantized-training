import struct
from dataclasses import dataclass, field
from typing import List

@dataclass
class SimplifiedParams:
    INPUT_OFFSET: int
    WEIGHT_OFFSET: int
    OUTPUT_OFFSET: int
    WEIGHT_TRANSPOSE: bool
    loops: List[List[int]] = field(default_factory=lambda: [[0]*6, [0]*6])
    inputXLoopIndex: List[int] = field(default_factory=lambda: [0]*2)
    inputYLoopIndex: List[int] = field(default_factory=lambda: [0]*2)
    reductionLoopIndex: List[int] = field(default_factory=lambda: [0]*2)
    weightLoopIndex: List[int] = field(default_factory=lambda: [0]*2)
    fxIndex: int = 0
    fyIndex: int = 0
    weightReuseIndex: List[int] = field(default_factory=lambda: [0]*2)
    STRIDE: int = 0
    REPLICATION: bool = False
    RELU: bool = False
    BIAS: bool = False
    BIAS_OFFSET: int = 0
    RESIDUAL: bool = False
    RESIDUAL_OFFSET: int = 0
    MAXPOOL: bool = False
    AVGPOOL: bool = False
    WEIGHT: bool = False
    STORE_IN_ACC: bool = False
    ACC_FROM_ACC: bool = False
    SOFTMAX: bool = False
    ATTENTION_SCALING: bool = False
    FC: bool = False
    NO_NORM: bool = False
    SOFTMAX_GRAD: bool = False
    FC_GRAD: bool = False
    NO_NORM_GRAD: bool = False
    RELU_GRAD: bool = False
    BIAS_GRAD: bool = False
    CROSS_ENTROPY_GRAD: bool = False
    MSE_GRAD: bool = False
    BCE_WITH_LOGITS_GRAD: bool = False
    INPUT_TRANSPOSE: bool = False
    CONCAT_INPUT: bool = False
    CONCAT_WEIGHT: bool = False
    SPLIT_OUTPUT: bool = False
    GRAD_CLIPPING: bool = False
    GRAD_CLIPPING_UNIT_TEST: bool = False
    WEIGHT_SPLITTING: bool = False
    WEIGHT_RESIDUAL_OFFSET: int = 0
    learningRate: float = 0.0
    ACC_T_INPUT: bool = False
    ACC_T_WEIGHT: bool = False
    ACC_T_OUTPUT: bool = False
    ACC_T_RESIDUAL: bool = False
    outputExpBias: int = 0
    WEIGHT_UPDATE: bool = False
    ERROR_FEEDBACK: bool = False
    depthwise: bool = False
    name: str = ""
    ATTENTION_MASK: bool = False
    MERGE_LORA_WEIGHT: bool = False
    QUANTIZE_TO_P8: bool = False
    MAX_REDUCE: bool = False
    ELWISE_ADD: bool = False
    NOP: bool = False

def pack_simplified_params(params: SimplifiedParams) -> bytes:
    # Define the format string for struct.pack
    format_string = (
        'iii?' + '6i6i' + '2i' * 4 + 'ii2i?' +
        '???i?' + '??' + '?????' + '?????' +
        'f' + '????' + 'i?' + '256s' + '???' + 'i'
    )
    
    # Pack the data into a binary format
    packed_data = struct.pack(
        format_string,
        params.INPUT_OFFSET,
        params.WEIGHT_OFFSET,
        params.OUTPUT_OFFSET,
        params.WEIGHT_TRANSPOSE,
        *params.loops[0], *params.loops[1],
        *params.inputXLoopIndex,
        *params.inputYLoopIndex,
        *params.reductionLoopIndex,
        *params.weightLoopIndex,
        params.fxIndex,
        params.fyIndex,
        *params.weightReuseIndex,
        params.STRIDE,
        params.REPLICATION,
        params.RELU,
        params.BIAS,
        params.BIAS_OFFSET,
        params.RESIDUAL,
        params.RESIDUAL_OFFSET,
        params.MAXPOOL,
        params.AVGPOOL,
        params.WEIGHT,
        params.STORE_IN_ACC,
        params.ACC_FROM_ACC,
        params.SOFTMAX,
        params.ATTENTION_SCALING,
        params.FC,
        params.NO_NORM,
        params.SOFTMAX_GRAD,
        params.FC_GRAD,
        params.NO_NORM_GRAD,
        params.RELU_GRAD,
        params.BIAS_GRAD,
        params.CROSS_ENTROPY_GRAD,
        params.MSE_GRAD,
        params.BCE_WITH_LOGITS_GRAD,
        params.INPUT_TRANSPOSE,
        params.CONCAT_INPUT,
        params.CONCAT_WEIGHT,
        params.SPLIT_OUTPUT,
        params.GRAD_CLIPPING,
        params.GRAD_CLIPPING_UNIT_TEST,
        params.WEIGHT_SPLITTING,
        params.WEIGHT_RESIDUAL_OFFSET,
        params.learningRate,
        params.ACC_T_INPUT,
        params.ACC_T_WEIGHT,
        params.ACC_T_OUTPUT,
        params.ACC_T_RESIDUAL,
        params.outputExpBias,
        params.WEIGHT_UPDATE,
        params.ERROR_FEEDBACK,
        params.depthwise,
        params.name.encode('utf-8'),
        params.ATTENTION_MASK,
        params.MERGE_LORA_WEIGHT,
        params.QUANTIZE_TO_P8,
        params.MAX_REDUCE,
        params.ELWISE_ADD,
        params.NOP
    )

    return packed_data

def map_operation(op) -> SimplifiedParams:
    pass