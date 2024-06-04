from .mapping_utils import OP_TO_MAPPING_FUNC


def _map_all_ops(node, args, output_dir):
    for mapping_func in OP_TO_MAPPING_FUNC.values():
        param = mapping_func(node, args, output_dir)
        if param is not None:
            return param
    return None


def map_operation(op, name, output_dir):
    params = []
    for node, args in zip(op.nodes, op.all_input_nodes):
        param = _map_all_ops(node, args, output_dir)
        if param is not None:
            params.append(param)
        else:
            print(f"Unsupported operation {node.name}: {node.target}")

    if len(params) == 0:
        return None

    from .param_pb2 import AcceleratorParam, VectorParam, PoolingParam, ReduceParam, ShapeParam

    param = AcceleratorParam()
    param.name = name

    if params[0].opcode in ["conv2d", "linear", "matmul"]:
        param.matrix_param.CopyFrom(params[0])
        if len(params) > 1:
            param.vector_params.extend(params[1:])
            param.fused = True

    if params[0].opcode == "layer_norm":
        assert len(params) == 1, "LayerNorm operation cannot be fused with other operations"
        param.matrix_param.CopyFrom(params[0])

    if isinstance(params[0], VectorParam):
        param.vector_params.extend(params)

    if isinstance(params[0], ReduceParam):
        assert len(params) == 1, "Reduce operation cannot be fused with other operations"
        param.reduce_param.CopyFrom(params[0])

    if isinstance(params[0], ShapeParam):
        assert len(params) == 1, "Shape operation cannot be fused with other operations"
        param.shape_param.CopyFrom(params[0])

    if isinstance(params[0], PoolingParam):
        assert len(params) == 1, "Pooling operation cannot be fused with other operations"
        param.pooling_param.CopyFrom(params[0])

    return param
