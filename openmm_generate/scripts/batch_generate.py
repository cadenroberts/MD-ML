#!/usr/bin/env python3
import os
import time
import traceback
from module import function
from module import preprocess
from module import simulation
from openmm.app import PDBFile
import openff.units

# Log capture
# From https://stackoverflow.com/questions/1218933/can-i-redirect-the-stdout-into-some-sort-of-string-buffer
import sys
from io import StringIO
import glob

class RedirectOutputs:
    def __init__(self):
        self._stdout = None
        self._stderr = None
        self._string_io = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = sys.stderr = self._string_io = StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def __str__(self):
        return self._string_io.getvalue()

def parse_integrator_params(integrator_params):
    # Parse the integrator params dict into apropreate units
    result = dict()
    # Convert entries in the form "1*atm" or "1/picosecond" into OpenMM quantities
    for k,v in integrator_params.items():
        # Pretend numbers are strings
        v = str(v)
        # Split on the first math operator, either * or /
        value_mul = v.split("*",1)
        value_div = v.split("/",1)

        def to_number(value):
            if "." in value:
                return float(value)
            else:
                return int(value)

        if len(value_mul[0]) < len(value_div[0]):
            value, units = value_mul
            value = to_number(value)
            value *= openff.units.openmm.string_to_openmm_unit(units)
        elif len(value_div) == 2:
            value, units = value_div
            value = to_number(value)
            value /= openff.units.openmm.string_to_openmm_unit(units)
        else: # No units
            value = to_number(value)
        result[k] = value
    return result

def prepare_one(pdbid, data_dir=None, input_dir=None, force=False, remove_ligands=False, implicit_solvent=False):
    if data_dir:
        function.set_data_dir(data_dir)

    finished_file_path = function.get_data_path(f'{pdbid}/processed/finished.txt')

    # FIXME: Should this also check if implicit_solvent has changed?
    if os.path.exists(finished_file_path):
        if force:
            os.remove(finished_file_path)
        else:
            print("Skipping prepare", pdbid, "(already prepared)")
            # FIXME: It would be better to check the contents of finished.txt instead of always returning True
            return True

    print("Processing", pdbid)

    t0 = time.time()
    ok = True
    with RedirectOutputs() as log:
        old_openmp_value = os.environ.get("OMP_NUM_THREADS", None)
        try:
            # The sqm program used by OpenMM to parameterize ligans makes very inefficient use of threads
            os.environ["OMP_NUM_THREADS"]="2"
            if input_dir:
                prepare_input = os.path.join(input_dir, pdbid + ".pdb")
            else:
                prepare_input = pdbid
            preprocess.prepare_protein(prepare_input, remove_ligands=remove_ligands, implicit_solvent=implicit_solvent)
        except Exception as e:
            ok = False
            traceback.print_tb(e.__traceback__)
            print(e)
        finally:
            if old_openmp_value:
                os.environ["OMP_NUM_THREADS"]=old_openmp_value
            else:
                del os.environ["OMP_NUM_THREADS"]

    with open(function.get_data_path(f'{pdbid}/processed/{pdbid}_process.log'),"wb") as f:
        f.write(str(log).encode("utf-8"))

    t1 = time.time() - t0
    finished_str = f"{pdbid} {('error', 'ok')[int(ok)]} ({round(t1,4)} seconds)"

    with open(function.get_data_path(f'{pdbid}/processed/finished.txt'), "w", encoding="utf-8") as finished_file:
        finished_file.write(finished_str)
    print(" ", finished_str)

    return ok

def simulate_one(pdbid, data_dir=None, input_dir=None, steps=10000, report_steps=1, prepare=False, remove_ligands=False,
                 prepare_implicit=False, force=False, timeout=None, integrator_params=None):
    # print("simulate_one:", pdbid, data_dir, steps, report_steps, prepare, force, timeout)
    interrupt_callback = None
    if timeout:
        interrupt_callback = lambda timeout=timeout : timeout > time.time()
        if not interrupt_callback():
            print("Canceled", pdbid, "(timeout)")
            return

    if data_dir:
        function.set_data_dir(data_dir)

    if prepare:
        #TODO: Split force prepare / force simulate into separate flags?
        if not prepare_one(pdbid, data_dir, input_dir, force, remove_ligands=remove_ligands, implicit_solvent=prepare_implicit):
            return

    finished_file_path = function.get_data_path(f'{pdbid}/simulation/finished.txt')
    continue_file_path = function.get_data_path(f'{pdbid}/simulation/continue.txt')

    if os.path.exists(finished_file_path):
        if force:
            os.remove(finished_file_path)
        else:
            print("Skipping", pdbid, "(already finished)")
            return

    should_continue = False
    action_name = "Simulating"
    if os.path.exists(continue_file_path):
        os.remove(continue_file_path)
        should_continue = True
        action_name = "Continuing"

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(action_name, pdbid, "on gpu", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print(action_name, pdbid)

    t0 = time.time()
    ok = True
    with RedirectOutputs() as log:
        try:
            pdb_path = function.get_data_path(f'{pdbid}/processed/{pdbid}_processed.pdb')
            atom_indices = function.get_non_water_atom_indexes(PDBFile(pdb_path).getTopology())
            steps_run = simulation.run(pdbid, pdb_path, steps, report_steps=report_steps, atomSubset=atom_indices,
                                       resume_checkpoint=should_continue, interrupt_callback=interrupt_callback,
                                       integrator_params=integrator_params)
        except Exception as e:
            ok = False
            traceback.print_tb(e.__traceback__)
            print(e)

    simulation_log_path = function.get_data_path(f'{pdbid}/simulation/{pdbid}_simulation.log')
    simulation_log_mode = "w"
    if should_continue:
        simulation_log_mode = "a"

    try:
        with open(simulation_log_path, simulation_log_mode, encoding="utf-8") as f:
            f.write(str(log))
    except Exception as e:
        print("!!! Failed to write log:", e)
        print(log)

    t1 = time.time() - t0
    if ok and steps_run != steps:
        # If fewer than the requested number of steps ran but there was no exception then
        # the simulation was gracefully interrupted and we can continue it later.
        finished_str = f"{pdbid} interrupted ({round(t1,4)} seconds)"
        with open(continue_file_path, "w", encoding="utf-8") as continue_file:
            continue_file.write(f"{steps_run}")
    else:
        finished_str = f"{pdbid} {('error', 'ok')[int(ok)]} ({round(t1,4)} seconds)"
        with open(finished_file_path, "w", encoding="utf-8") as finished_file:
            finished_file.write(finished_str)
    print(" ", finished_str)

def init_on_gpu(gpu_list, counter):
    gpu_id = None
    with counter.get_lock():
        gpu_id = gpu_list[counter.value % len(gpu_list)]
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

def main():
    import json
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("pdbid_list", type=str, nargs="*", help="The PDB ids to process, or .json file containing an array of PDB ids")
    parser.add_argument("--batch-size", default=None, type=int, help="Split pdbid_list into batches of this size")
    parser.add_argument("--batch-index", default=0, type=int, help="If splitting into batches, select which batch to run")
    parser.add_argument("-f", "--force", action='store_true', help="Force simulate (and prepare if enabled) to run even the requested pdbids have already finished")
    parser.add_argument("--prepare", action='store_true', help="Run prepare if the system has not already been set up")
    parser.add_argument("--prepare-implicit", action='store_true', help="Run prepare with an implicit solvent model if the system has not already been set up")
    parser.add_argument("--remove-ligands", action='store_true', help="Remove ligands instead of parameterizing in prepare")
    parser.add_argument("--integrator", default=None, type=str, help="A json file specifying the integrator parameters")
    parser.add_argument("--pool-size", default=10, type=int, help="Number of simultaneous simulations to run")
    parser.add_argument("--steps", default=10000, type=int, help="Total number of steps to run")
    parser.add_argument("--report-steps", default=1, type=int, help="Save data every n-frames")
    parser.add_argument("--data-dir", default="../data/", type=str, help="Output data directory")
    parser.add_argument("--input-dir", default=None, type=str, help="Input data directory, if set PDB files will be copied from here instead of download from RCSB")
    parser.add_argument("--gpus", default=None, type=str, help="A comma delimited lists of GPUs to use e.g. '0,1,2,3'")
    parser.add_argument("--timeout", default=None, type=float, help="The maximum time to run in hours (e.g. 0.5 = 30 minutes)")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.data_dir):
        os.makedirs(os.path.realpath(args.data_dir), exist_ok=True)

    pdbid_list = args.pdbid_list

    if not pdbid_list:
        if not args.input_dir:
            print("Either an input directory of a list of pdbs must be given")
            return 1
        else:
            pdbid_list = []
            for i in glob.glob(os.path.join(args.input_dir, "*.pdb")):
                pdbid_list.append(os.path.splitext(os.path.basename(i))[0])
            pdbid_list = sorted(pdbid_list)
            if not pdbid_list:
                print(f"Could not find any pdbs in \"{args.input_dir}\"")
                return 1

    if pdbid_list[0].endswith(".json"):
        print("Taking inputs from list:", args.pdbid_list[0])
        with open(args.pdbid_list[0], "r") as f:
            pdbid_list = json.load(f)

    if args.prepare_implicit:
        args.prepare = True

    if args.integrator:
        with open(args.integrator, "r") as f:
            integrator_params = json.load(f)
        integrator_params = parse_integrator_params(integrator_params)
    else:
        integrator_params = None

    try:
        multiprocessing.set_start_method('spawn') # because NERSC says to use this one?
    except Exception as e:
        print("Multiprocessing:", e)

    if args.batch_size:
        batch_pdbid_list = pdbid_list[args.batch_index*args.batch_size:(args.batch_index+1)*args.batch_size]
    else:
        batch_pdbid_list = pdbid_list

    print(batch_pdbid_list)

    init_function = None
    init_args = None
    if args.gpus is not None:
        gpu_list = [int(i) for i in args.gpus.split(",")]
        init_args = (gpu_list, multiprocessing.Value('i', 0, lock=True))
        init_function = init_on_gpu

    if args.timeout:
        timeout = args.timeout * 3600 + time.time()
    else:
        timeout = None

    t0 = time.time()
    with multiprocessing.Pool(args.pool_size, initializer=init_function, initargs=init_args) as pool:
        pending_results = []
        for pdbid in batch_pdbid_list:
            kwargs_dict = {"data_dir":args.data_dir, "input_dir":args.input_dir, "steps":args.steps,
                           "report_steps":args.report_steps, "prepare":args.prepare,
                           "prepare_implicit":args.prepare_implicit,
                           "remove_ligands": args.remove_ligands,
                           "force":args.force, "timeout":timeout,
                           "integrator_params":integrator_params}
            pending_results += [pool.apply_async(simulate_one,
                                                 (pdbid,), kwargs_dict)]
        
        while pending_results:
            pending_results = [i for i in pending_results if not i.ready()]
            if pending_results:
                pending_results[0].wait(1)
    
    t1 = time.time() - t0
    print(f"Finished {len(batch_pdbid_list)} in {round(t1,4)} seconds")

    return 0

if __name__ == "__main__":
    sys.exit(main())
