#!/usr/bin/env python3
import subprocess
import argparse
import os
import concurrent.futures
import traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multiple simulate.py processes in parallel. Any unparsed arguments are passed to simulate.py. Use -- to escape arguments understood by both tools."
        )
    parser.add_argument("pdbs", nargs="+", help="List of PDB ids to run")
    parser.add_argument("-m", "--model", required=True, type=str, help="Path to the model, if a directory is given checkpoint-best.pth will be used")
    parser.add_argument("-d", "--input-dir", type=str, default="./", help="Path to input data (pdbs or prepared cg data)")
    parser.add_argument("-o", "--output", default="./", help="Output directory")
    parser.add_argument("-f", "--format", default="h5", help="Output format (h5 or pdb)")

    args, unkown_args = parser.parse_known_args()

    model_path = args.model
    if (os.path.isdir(model_path)):
        model_path = os.path.join(model_path, "checkpoint-best.pth")
    assert os.path.exists(model_path), f"Model path does not exist: {model_path}"

    output_prefix = os.path.basename(os.path.dirname(model_path))
    output_prefix = "sim_" + output_prefix.removeprefix("model_")
    output_prefix = os.path.join(args.output, output_prefix)
    output_format = args.format
    assert output_format in ["h5", "pdb"]

    input_dir = args.input_dir
    # TODO: If input_dir is cg data directory assert that the priors match

    simulate_args = []
    simulate_args = ["--steps=50000",  "--save-steps=50", "--replicas=5", "--max-num-neighbors=128"]
    simulate_args.extend(unkown_args)

    def run_simulation(input_dir, pos, pdbid, output_prefix, output_format, args):
        # TODO: Capture output to a logfile (maybe tee stderr show the progress bars?)
        input_fn = os.path.join(input_dir, pdbid + ".pdb")
        try:
            if not os.path.exists(input_fn):
                input_fn = os.path.join(input_dir, pdbid)
                if not os.path.isdir(input_fn):
                    raise RuntimeError(f"Input file not found: {input_fn}")

            output_name = output_prefix + f"_{pdbid}.{output_format}"
            call_args = ["./simulate.py", model_path, input_fn, "-o", output_name] + args
            # print(call_args)

            call_env = os.environ.copy()
            call_env["TQDM_MINITERS"] = "10"
            call_env["TQDM_POSITION"] = f"{pos}"
            call_env["TQDM_BAR_FORMAT"] = pdbid + ": {percentage:3.0f}% ({n_fmt}/{total_fmt}) ({elapsed}<{remaining}, {rate_fmt}{postfix})"

            print("Starting simulation:", " ".join(call_args))
            subprocess.run(call_args, env=call_env, check=True)
        except Exception as e:
            print(f"!!! Simulation of \"{input_fn}\" failed.")
            traceback.print_tb(e.__traceback__)
            print(e)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, pdbid in enumerate(args.pdbs):
            executor.submit(run_simulation, input_dir, i, pdbid, output_prefix, output_format, simulate_args)
