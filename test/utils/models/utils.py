def get_transform_args(args, vector_stages):
    transform_args = {
        "patterns": vector_stages,
        "transpose_weight": args.transpose_weight,
        "transpose_fc": args.transpose_fc,
        "conv2d_padding": args.padding,
        "cache_size": args.cache_size,
        "num_banks": args.num_banks,
        "block_size": args.block_size,
        "perform_tiling": args.perform_tiling,
    }
    return transform_args

def get_compile_args(args):
    compile_args = {
        "cache_size": args.cache_size,
        "bank_width": args.bank_width,
        "bank_size": None if args.cache_size is None else args.cache_size // args.num_banks,
        "output_dir": args.model_output_dir,
        "output_file": args.model,
    }
    return compile_args