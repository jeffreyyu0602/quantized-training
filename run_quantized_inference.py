import subprocess
import sys

dtypes = ["posit8_1", "posit8_2", "e4m3"]

operations = [
    "gemm,residual,norm,activation,scaling",
    "gemm,residual,norm,activation",
    "gemm,residual,norm",
    "gemm,residual",
    "gemm",
]


def main():
    for dtype in dtypes:
        for ops in operations:
            command = ['python'] + sys.argv[1:]
            command += [
                "--dtype", dtype, "--quantize_fwd", ops, "--quantize_weights"
            ]
            print("Running:", ' '.join(command))
            subprocess.run(command, check=True)

    print("All commands executed.")


if __name__ == "__main__":
    main()
