#!/usr/bin/env python3
from gen_benchmark import Benchmark300, machines, runReport, ModelPath, ComponentAnalysisTypes
import pathlib
import argparse
import os
from pathlib import Path
import re

def sort_key(filepath):
    # Extract the base filename
    filename = filepath.split('/')[-1]
    # Match numerical checkpoint values if they exist
    match = re.match(r'checkpoint-(\d+)\.pth', filename)
    if match:
        return int(match.group(1))  # Return the numerical value for sorting
    elif filename == 'checkpoint-best.pth':
        return float('inf')  # Push "best" to the end
    elif filename == 'checkpoint-mini.pth':
        return float('inf') - 1  # Push "mini" just before "best"
    return float('inf') - 2  # Push "checkpoint.pth" just before "mini"


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_path", help="The path of the model to benchmark")
    arg_parser.add_argument("--temperature", type=int, help="Temperature in Kelvin of the model")
    arg_parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=False, help="Use caching the tica models")
    arg_parser.add_argument("--machine", type=str, required=True, choices=machines.keys(), help="Which server this is being run on")
    arg_parser.add_argument("--proteins", type=str, default=None, help="Proteins to run benchmark on, default is all of them", nargs="+")
    arg_parser.add_argument("--output-dir", type=str, default=None, help="Output directory of benchmarks")
    arg_parser.add_argument("--start", type=int, default=0, help="Start from a specific checkpoint, default is 0")
    arg_parser.add_argument("--end", type=int, default=-1, help="End at a specific checkpoint, default is -1 which means the last checkpoint")
    arg_parser.add_argument("--disable-wandb", action=argparse.BooleanOptionalAction, default=False, help="Disable wandb logging")
    args = arg_parser.parse_args()

    # put the code below into a separate function
    # prepBenchmark(args.model_path, args.temperature, args.use_cache, args.machine, args.proteins, args.output_dir)

    model_path = Path(args.model_path)
    checkpoints = sorted(model_path.glob("*.pth"))
    checkpoints = [f for f in checkpoints if str(f).count('-') < 2]
    checkpoints = sorted(checkpoints, key=sort_key)
    checkpoints = [checkpoints[i] for i in range(args.start, args.end if args.end != -1 else len(checkpoints))]
    print('Will benchmark the following checkpoints', checkpoints)
    print('It generally takes 1h to benchmark 6 proteins')
    print('Estimated time: %d hours' % (len(checkpoints) * 1.0/6 * (len(args.proteins))))
    # adsd

    benchmarks = []
    global_out_dir = Path(args.output_dir + '_all_checkpoints')
    for c, checkpoint in enumerate(checkpoints):
        checkpoint_name: str = os.path.basename(checkpoint).split(".")[0]

        to_benchmark = ModelPath(checkpoint, False, None, 100000, 1000, 20)
        
        if c == 0:
            benchmarks.append(Benchmark300(to_benchmark, args.use_cache, args
                                           .machine, args.proteins, args.output_dir, False, ComponentAnalysisTypes.TICA, False))
            

            benchmarks[c].output_dir = os.path.join(global_out_dir, checkpoint_name)
            pathlib.Path(benchmarks[c].output_dir).mkdir(parents=True)
            benchmarks[c].log_dir = benchmarks[c].output_dir

        else:
            benchmarks.append(Benchmark300(to_benchmark, args.use_cache, args
                                           .machine, args.proteins, global_out_dir.joinpath(checkpoint_name), False, ComponentAnalysisTypes.TICA, False))
              
        benchmarkFile = benchmarks[c].runParallel()        
        runReport(benchmarkFile,
                  do_kl_divergence=True,
                  also_plot_locally=True,
                  do_rmsd_metrics=False,
                  disable_wandb=args.disable_wandb,
                  plot_individuals=False)


if __name__ == "__main__":
    main()
