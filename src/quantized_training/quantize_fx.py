import time
import math

import torch
import torch.nn as nn
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.export import export
from torch.fx import GraphModule

from transformers import AutoModelForSequenceClassification

from quantized_training import quantize_to_posit, add_training_args, quantize_model, get_default_qconfig

    
class Observer(torch.ao.quantization.FakeQuantizeBase):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = quantize_to_posit(x, 8, 1)
        return x
    
    def calculate_qparams(self, **kwargs):
        raise NotImplementedError

def quantize_fx(model: GraphModule, patterns):
    # Go through all the nodes in the Graph
    for node in model.graph.nodes:
        if any(str(node.target) == pattern for pattern in patterns):
            new_args = []
            for i, arg in enumerate(node.args):
                if str(node.target) == "aten.addmm.default" and (i == 0 or i == 2):
                    new_args.append(arg)
                    continue
                prefix = 'activation_pre_process_'
                get_new_obs_or_fq_name = get_new_attr_name_with_prefix(prefix)
                obs_or_fq_name = get_new_obs_or_fq_name(model)
                obs_or_fq = Observer(node.name)
                setattr(model, obs_or_fq_name, obs_or_fq)
                with model.graph.inserting_after(arg):
                    new_node = model.graph.call_module(obs_or_fq_name, (arg,))
                new_args.append(new_node)
            node.args = tuple(new_args)

    return GraphModule(model, model.graph)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased").eval()
    example_inputs = (torch.randint(0, 200, (1, 128)),)

    parser = add_training_args()
    args = parser.parse_args()
    quantized_model = quantize_model(model, args, inplace=False)

    start_time = time.time()
    orig_output = quantized_model(*example_inputs)
    time_taken = time.time() - start_time
    print(f"Original model inference takes: {time_taken} seconds")

    print("Original output")
    print(orig_output)

    start_time = time.time()
    exported_program: torch.export.ExportedProgram = export(
        model, args=example_inputs, kwargs={}
    )
    # print(exported_program.graph.print_tabular())
    time_taken = time.time() - start_time
    print(f"torch.export takes: {time_taken} seconds")

    start_time = time.time()
    patterns = set(["aten.addmm.default", "aten.bmm.default"])
    exported_program = quantize_fx(exported_program.module(), patterns)
    time_taken = time.time() - start_time
    print(f"quantize_fx takes: {time_taken} seconds")

    # print(exported_program.graph.print_tabular())

    start_time = time.time()
    new_output = exported_program(*example_inputs)
    time_taken = time.time() - start_time
    print(f"GraphModule inference takes: {time_taken} seconds")

    print("New output")
    print(new_output)