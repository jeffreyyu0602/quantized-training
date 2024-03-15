import time

import torch
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix, assert_and_get_unique_device
from torch.export import export
from torch.fx import GraphModule
from functorch.compile import aot_function

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM

from quantized_training import quantize_to_posit, add_training_args, quantize

QUANTIZATION_OPERATORS = {
    "gemm": [
        (torch.ops.aten.addmm.default, (1,)),
        (torch.ops.aten.mm.default, (0,)),
        torch.ops.aten.bmm.default,
        torch.ops.aten.convolution.default,
    ],
    "activation": [
        (torch.ops.aten._softmax.default, (0,)),
        (torch.ops.aten.gelu, (0,)),
        (torch.ops.aten.relu.default, (0,)),
    ],
    "norm": [(torch.ops.aten.native_layer_norm.default, (0,))],
    "residual": [torch.ops.aten.add.Tensor],
    "scaling": [torch.ops.aten.div.Tensor],
}

class FakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        print("FakeQuantizeFunction forward")
        return quantize_to_posit(x, 8, 1)

    @staticmethod
    def backward(ctx, grad_output):
        print("FakeQuantizeFunction backward")
        return grad_output

class Observer(torch.ao.quantization.FakeQuantizeBase):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FakeQuantizeFunction.apply(x)

    def calculate_qparams(self, **kwargs):
        raise NotImplementedError

def prepare(model: GraphModule, patterns):
    named_modules = {}
    # Go through all the nodes in the Graph
    for node in model.graph.nodes:
        matching_pattern = None
        for pattern in patterns:
            if node.target == (pattern[0] if isinstance(pattern, tuple) else pattern):
                matching_pattern = pattern
                break

        if matching_pattern is not None:
            new_args = []
            for i, arg in enumerate(node.args):
                if isinstance(matching_pattern, tuple) and i not in matching_pattern[1]:
                    new_args.append(arg)
                    continue
                obs_or_fq = Observer(node.name)
                model_device = assert_and_get_unique_device(model)
                if model_device:
                    obs_or_fq.to(model_device)
                # add obs_or_fq module as attribute
                prefix = 'activation_pre_process_'
                get_new_obs_or_fq_name = get_new_attr_name_with_prefix(prefix)
                obs_or_fq_name = get_new_obs_or_fq_name(model)
                setattr(model, obs_or_fq_name, obs_or_fq)
                named_modules[obs_or_fq_name] = obs_or_fq
                with model.graph.inserting_after(arg):
                    new_node = model.graph.call_module(obs_or_fq_name, (arg,))
                new_args.append(new_node)
            node.args = tuple(new_args)
    return GraphModule(model, model.graph)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").train()
    # example_inputs = (torch.randint(0, 200, (1, 128)),)
    # example_labels = torch.randint(0, 2, (1,))

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        attn_implementation="eager"
    ).train()
    example_inputs = (torch.randint(0, 200, (1, 1024)),)
    example_labels = example_inputs[0].clone()

    parser = add_training_args()
    args = parser.parse_args()
    quantized_model = quantize(model, args, inplace=False)

    start_time = time.time()
    orig_output = quantized_model(example_inputs[0])
    time_taken = time.time() - start_time
    print(f"Original model inference takes: {time_taken} seconds")

    print("Original output")
    print(orig_output)

    start_time = time.time()
    exported_program: torch.export.ExportedProgram = export(
        model, args=example_inputs, kwargs={'labels': example_labels}
    )
    time_taken = time.time() - start_time
    print(f"torch.export takes: {time_taken} seconds")

    # print(exported_program.graph.print_tabular())

    ops = [op.lower() for op in args.quantize_fwd.split(',')] if args.quantize_fwd is not None else []
    patterns = tuple(mod for op in ops for mod in QUANTIZATION_OPERATORS[op])

    start_time = time.time()
    exported_program = prepare(exported_program.module(), patterns)
    time_taken = time.time() - start_time
    print(f"prepare takes: {time_taken} seconds")

    print(exported_program.graph.print_tabular())

    start_time = time.time()
    new_output = exported_program(example_inputs[0], example_labels)
    time_taken = time.time() - start_time
    print(f"GraphModule inference takes: {time_taken} seconds")

    print("New output")
    print(new_output)

    loss = new_output.loss
    loss.backward()