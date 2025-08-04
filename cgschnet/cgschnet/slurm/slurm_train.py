#!/usr/bin/env python3
import subprocess
import argparse
import os
import shlex
import tempfile
import platform

# Note: The base python3 environment on perlmutter.nersc.gov is 3.6.15 as of 9/10/2024

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enqueue a train.py job using sbatch. Any unparsed arguments are passed to train.py. Use -- to escape arguments understood by both tools."
        )
    parser = argparse.ArgumentParser(description="Train a CGSchNet network")
    parser.add_argument("input", help="Processed data to train on")
    parser.add_argument("result", help="Checkpoint directory to save to")
    # TODO: Allow this to default to the currently active environment?
    parser.add_argument("--env-python", required=True, help="The python interpreter use to run train.py")
    parser.add_argument("--email", required=True, help="The email to send slurm messages to")
    parser.add_argument("--timeout", default="2:00:00", help="Slurm job timeout in HH:MM:SS format")
    parser.add_argument("--no-sbatch", default=False, action='store_true', help="Generate the sbatch script but don't enqueue it")

    args, unknown_args = parser.parse_known_args()

    # Find this file's location and train.py relative to it
    assert __file__
    script_working_dir = os.path.dirname(__file__)
    train_script_path = os.path.abspath(os.path.join(script_working_dir, "..", "scripts", "train.py"))
    assert os.path.exists(train_script_path)
    print("Training script:", train_script_path)

    # Ensure we can find the environment used to run train.py
    assert os.path.exists(args.env_python)

    # Set the cluster specific parameters
    template_path = "train.slurm.template"
    login_node_gpus = "--gpus=0"
    if "delta.ncsa.illinois.edu" in platform.node():
        print("Current cluster: Delta")
        template_path = "train.delta.template"
        login_node_gpus = "--gpus=cpu"
    else:
        print("Current cluster: NERSC")
        # Assume NERSC

    # Load the script template
    template_path = os.path.join(script_working_dir, template_path)
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    assert os.path.isdir(args.input), "Input directory does not exist!"

    # Escape the arguments to ensure we can safely put them back in a shell script
    training_args = [shlex.quote(i) for i in unknown_args]
    input_path_arg = shlex.quote(args.input)
    result_path_arg = shlex.quote(args.result)

    # Verify the output directory exists or create it
    if os.path.exists(args.result):
        # If the directory already exists use "--dry-run" so we don't modify anything
        print("--- Output directory exists, validating settings ---")
        run_args = [args.env_python, train_script_path, input_path_arg, result_path_arg] + training_args + ["--epochs=0", login_node_gpus, "--dry-run"]
        subprocess.run(run_args, check=True)
        print("--- Done ---")
    else:
        # Otherwise initialize it but don't try to actually load any data (other people may be using the login node GPUs)
        print("--- Initializing output directory ---")
        run_args = [args.env_python, train_script_path, input_path_arg, result_path_arg] + training_args + ["--epochs=0", login_node_gpus]
        subprocess.run(run_args, check=True)
        print("--- Done ---")

    # Construct a command line for train.py, adding appropriate GPU values for perlmutter
    # Each node has 4 gpus: --gpus="0,1,2,3"
    # Each A100 has at least 40GB of ram (so 35000*4 atoms total): --apc=140000
    slurm_training_args = [args.env_python, train_script_path, input_path_arg, result_path_arg, "--gpus=0,1,2,3", "--apc=140000"] + training_args

    #FIXME: Should we assert that the batch size is at least 4? If not we can't use all the GPUs and the apc value will be wrong.
    # It should probably always be a multiple of 4 to optimally use the GPUs & ensure --apc is safe

    # Generate a temporary slurm script
    template = template.replace("TEMPLATE_SLURM_EMAIL", args.email)
    template = template.replace("TEMPLATE_SLURM_JOB_NAME", "train_" + args.email)
    template = template.replace("TEMPLATE_SLURM_TIMEOUT", args.timeout)
    template += "\n" + " ".join(slurm_training_args)
    print(template)
    print("---")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="train.", suffix=".slurm",
                                     dir="./", delete=(not args.no_sbatch)) as f:
        f.write(template)
        f.flush() # Ensure the data is actually in the file before we use it

        if args.no_sbatch:
            print("Script generated:", f.name)
        else:
            sbatch_command = ["sbatch", f.name]
            # sbatch_command = ["bash", f.name] # Run the script rather than enqueuing it
            print("Enqueuing:", " ".join(sbatch_command))
            subprocess.run(sbatch_command, check=True)
            # Future TODO: Parse the sbatch output, potentially queue another job