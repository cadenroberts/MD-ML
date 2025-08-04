#!/usr/bin/env python3

import json
import os
import sys
import glob
import shutil
import argparse

def check_files_match(file_a, file_b):
    with open(file_a, "r") as f:
        data_a = f.read()
    with open(file_b, "r") as f:
        data_b = f.read()
    return data_a == data_b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the model to add a prior to")
    parser.add_argument("--cg", default=None, help="Data path to copy prior from")
    parser.add_argument("-f", "--force", action="store_true", default=False, help="Overwrite the prior if the model already has one")
    args = parser.parse_args()

    model_path = args.model_path

    training_info_path = os.path.join(model_path, "training_info.json")
    assert os.path.exists(training_info_path), "Not a valid model directory"
    output_prior_path = os.path.join(model_path, "priors.yaml")
    if os.path.exists(output_prior_path) and not args.force:
        print("!!! Model already has a prior, use -f to overwrite")
        return 1

    if args.cg:
        input_directory = args.cg
        if not os.path.isdir(input_directory):
            print("!!! CG data directory does not exist!")
            return 1
    else:
        with open(training_info_path, "r", encoding="utf-8") as f:
            training_info = json.load(f)
        max_info_epoch = max(map(int, training_info.keys()))
        input_directory = training_info[str(max_info_epoch)]["input_directory"]

    print("Data directory:", input_directory)
    if not os.path.isdir(input_directory):
        print("!!! CG data directory does not exist or is not a directory, please use --cg=<path> to specify a path")
        return 1
    prior_path = os.path.join(input_directory, "priors.yaml")
    if not os.path.exists(prior_path):
        prior_path_list = glob.glob(os.path.join(input_directory, "*", "raw", "*_priors.yaml"))
        assert len(prior_path_list) > 0
        if len(prior_path_list) > 1:
            assert check_files_match(prior_path_list[0], prior_path_list[1])
        prior_path = prior_path_list[0]
    prior_params_path = prior_path.replace("priors.yaml", "prior_params.json")
    assert os.path.exists(prior_path)
    assert os.path.exists(prior_params_path)
    output_prior_params_path = os.path.join(model_path, "prior_params.json")
    print("Prior: ",prior_path)
    print("Params:",prior_params_path)
    shutil.copy(prior_path, output_prior_path)
    shutil.copy(prior_params_path, output_prior_params_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())