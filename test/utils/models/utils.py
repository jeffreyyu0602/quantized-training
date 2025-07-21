def get_transform_args(args, vector_stages):
    transform_args = {
        "patterns": vector_stages,
        "transpose_weight": args.transpose_weight,
        "transpose_fc": args.transpose_fc,
        "unroll_dimension": args.hardware_unrolling,
        "cache_size": args.cache_size,
    }
    return transform_args

def get_compile_args(args):
    compile_args = {
        "cache_size": args.cache_size,
        "bank_width": args.bank_width,
        "bank_size": (
            args.cache_size // args.num_banks
            if args.cache_size is not None and args.num_banks is not None else None
        ),
        "output_dir": args.model_output_dir,
        "output_file": args.model,
    }
    return compile_args