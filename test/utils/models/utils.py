def get_transform_args(args, vector_stages):
    transform_args = {
        "patterns": vector_stages,
        "transpose_weight": args.transpose_weight,
        "transpose_fc": args.transpose_fc,
        "conv2d_padding": args.padding,
    }
    return transform_args

def get_compile_args(args):
    compile_args = {
        "bank_width": args.bank_width,
        "bank_size": args.bank_size,
        "weight_persistent": args.weight_persistent,
        "output_dir": args.model_output_dir,
        "output_file": args.model,
    }
    return compile_args