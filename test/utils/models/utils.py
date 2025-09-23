def get_transform_args(args, vector_stages):
    transform_args = {
        "patterns": vector_stages,
        "transpose_weight": args.transpose_weight,
        "transpose_fc": args.transpose_fc,
        "unroll_dims": args.hardware_unrolling,
        "cache_size": args.cache_size,
        "num_banks": args.num_banks,
        "conv2d_im2col": args.conv2d_im2col,
        "fuse_reshape": (
            not args.dont_fuse_reshape
            and (args.hardware_unrolling is None or max(args.hardware_unrolling) < 64)
        ),
    }
    return transform_args

def get_compile_args(args):
    compile_args = {
        "cache_size": args.cache_size,
        "num_banks": args.num_banks,
        "bank_width": args.bank_width,
        "unroll_dims": args.hardware_unrolling,
        "output_dir": args.model_output_dir,
        "output_file": args.model,
    }
    return compile_args
