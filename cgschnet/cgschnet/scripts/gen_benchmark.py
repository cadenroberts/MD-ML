#!/usr/bin/env python3
import argparse
import preprocess
import os
import sys
import glob
import numpy
import mdtraj
from pathlib import Path
import random
import pickle
from typing import Optional
from dataclasses import dataclass, asdict
from report_generator.cache_loading import load_cache_or_make_new
from report_generator.tica_plots import DimensionalityReduction, TicaModel, PCAModel, generate_tica_model_from_scratch, generate_pca_model_from_scratch
from report_generator.traj_loading import load_native_trajs_stride, load_model_traj_pickle, NativeTrajPath, NativeTrajPathH5, NativeTrajPathNumpy, ModelTraj, load_model_traj, apply_cg
from report_generator.reaction_coordinate import get_reaction_coordinate_kde
from report_generator.contact_maps import get_contact_maps
from report_generator.bond_and_angle_analysis import get_bond_angles_cached
from report_generator.msm_analysis import do_msm_analysis, MsmRmsdStatistics
from gen_report import runReport
from westpa_analysis.westpa_helpers import extract_simulation_config, calculate_component_values
import subprocess
import json
import yaml

import threading
from multiprocessing.dummy import Pool as ThreadPool
import pickle

import torch
from enum import Enum

class ComponentAnalysisTypes(Enum):
    TICA = 1
    PCA = 2


import logging
logging.basicConfig(
    level=logging.DEBUG,
)

# On Delta, nodes have 256GB of RAM and will be killed due to going out of memory (if running on 6 proteins: bba chignolin homeodomain trpcage wwdomain proteinb). Proteinb consumes the highest RAM, around 200GB.
num_threads: int | None = None
gpu_list: list[int] | None = None
generate_trajs_semaphore: threading.Semaphore | None = None
available_gpus: list[bool] | None = None
find_gpu_mutex: Optional[threading.Lock] = None

def init_gpu_locks(gpu_ids: list[int]):
    # This is really ugly but I don't want to rock the boat too much right now - Daniel
    # FIXME: If you see this and want to replace it with something that doesn't use globals go for it
    global num_threads, gpu_list, generate_trajs_semaphore, available_gpus, find_gpu_mutex
    num_threads = len(gpu_ids)
    gpu_list = gpu_ids
    generate_trajs_semaphore = threading.Semaphore(len(gpu_ids))
    available_gpus = [True for _ in gpu_ids]
    find_gpu_mutex = threading.Lock()

# set the semaphore below to 1 when not striding in load_native_trajs_stride as the threads will run out of RAM memory. Loading the entire native trajs for homeodomain/proteinb takes around 200GB RAM.
load_trajs_semaphore = threading.Semaphore(6) 

# I think we'll have to discard the first frames in each model traj, as they are biasing the model KDE towards the native KDE. To see this effect, look on delta at the results with the following command:
#  imgcat /work/hdd/bbpa/benchmarks/000027_all_12368_cyrusc_081724/*.png (used 100k steps and saved every 1,000 steps
# Look at Model Points in TICA space, where R10 shows points after 10% of steps (I plotted it there the other way around) - and R90 shows all 20 replicas after 90,000 steps, which truly shows the equilibrium distribution of the model. 
# Raz later note: I did the above, it's in report.py I think

# total # of frames is 10,000. stride=10 means => 1000 frames * 859 starting points
NATIVE_PATHS_STRIDE = 100 # only take every N frames in the native trajectories

@dataclass
class MachineConf:
    data_300_path: Path
    data_350_path: Path
    cache_path: Path
    sims_store_dir: Path
    rmsd_dir: Path

@dataclass
class ModelPath:
    model_path: Path
    prior_only: bool
    prior_nn: Path | None
    num_steps: int
    num_save_steps: int
    trajs_per_protein: int

@dataclass
class OldBenchmarkRerun:
    old_benchmark_dir: Path

@dataclass
class TrajFolder:
    traj_folder: Path

@dataclass
class WestpaFolder:
    folder: Path
    
Benchmarkables = ModelPath | TrajFolder | OldBenchmarkRerun | WestpaFolder

machines: dict[str, MachineConf] = {
    "bizon": MachineConf(Path("/media/DATA_18_TB_1/andy/benchmark_set_5/"),
                         Path("/media/DATA_18_TB_1/daniel_s/majewski_2023_data"),
                         Path("/media/DATA_18_TB_1/andy/benchmark_cache_v2"),
                         Path("/media/DATA_18_TB_1/benchmark_sims"),
                         Path("/media/DATA_18_TB_1/andy/expiremental_structure")),
    "delta": MachineConf(Path("/work/hdd/bbpa/acbruce/md_data/generate"), # data at 300K
                         Path("/work/hdd/bbpa/acbruce/cecilia_numpy/majewski_2023_data"), # data 350K
                         Path("/work/hdd/bbpa/acbruce/benchmark_cache"), # cache path
                         Path("/work/hdd/bbpa/benchmarks"),
                         Path("/work/hdd/bbpa/acbruce/expiremental_structure"))} # sims_store_dir

@dataclass
class BenchmarkModelPath:
    checkpoint_path: Path | None
    model_folder: Path
    prior_only: bool
    prior_nn: Path | None
    num_steps: int
    num_save_steps: int
    trajs_per_protein: int

@dataclass
class BenchmarkTrajFolder:
    folder: Path
    traj_paths: list[Path]

@dataclass
class BenchmarkOldDir:
    folder: Path
    proteins_pickles: dict[str, Path]

@dataclass
class BenchmarkWestpaDir:
    folder: Path
    
class Benchmark:
    temperature: int
    native_paths: dict[str, list[NativeTrajPath]]
    starting_poses: dict[str, list[NativeTrajPath]]
    only_gen_cache: bool
    proteins: list[str]
    machine: MachineConf
    output_dir: Path
    log_dir: Path
    benchmark_descriptor: BenchmarkModelPath | BenchmarkTrajFolder | BenchmarkOldDir | BenchmarkWestpaDir
    component_analysis: ComponentAnalysisTypes
    make_table: bool
    def __init__(
            self,
            to_benchmark: Benchmarkables,
            use_cache: bool,
            machine: MachineConf,
            proteins: list[str],
            output_dir_c: Path | None,
            only_gen_cache: bool,
            component_analysis: ComponentAnalysisTypes,
            make_table: bool
    ) -> None:
        self.component_analysis = component_analysis
        self.make_table = make_table
        
        match to_benchmark:
            case ModelPath(model_path,
                           prior_only,
                           prior_nn,
                           num_steps,
                           num_save_steps,
                           trajs_per_protein):
                # if model path is a checkpoint, store the model path separately
                if model_path.suffix == ".pth":
                    checkpoint_path = model_path
                    real_model_path = model_path.parent
                else:
                    checkpoint_path = None
                    real_model_path = model_path
                self.benchmark_descriptor = BenchmarkModelPath(
                        checkpoint_path,
                        real_model_path,
                        prior_only,
                        prior_nn,
                        num_steps,
                        num_save_steps,
                        trajs_per_protein)
            case TrajFolder(trajs_folder):
                traj_paths = list(trajs_folder.iterdir())
                self.benchmark_descriptor = BenchmarkTrajFolder(trajs_folder, traj_paths)
            case OldBenchmarkRerun(old_dir):
                with open(os.path.join(old_dir, "benchmark.json"), "r") as f:
                    json_data=f.read()

                benchmark_json: dict = json.loads(json_data)
                proteins_dict: dict = benchmark_json["proteins"]
                
                self.benchmark_descriptor = BenchmarkOldDir(
                    old_dir,
                    {name: Path(value["gen_pickle_path"]) for name, value in proteins_dict.items()})
            case WestpaFolder(westpa_folder):
                self.benchmark_descriptor = BenchmarkWestpaDir(westpa_folder)
                

                
                
                
                    
        self.force_cache_regen = not use_cache
        self.machine = machine
        self.proteins = proteins
        self.only_gen_cache = only_gen_cache

        # If there's a trajs folder, we're not benchmarking a model, we're benchmarking trajectories


        # machine = machines[self.machine]
        if output_dir_c is not None:
            self.output_dir = output_dir_c 
        else:
            simNr = 1
            flds = list(self.machine.sims_store_dir.glob("0*"))
            if len(flds) > 0:
                simNr = max([int(f.parts[-1][:6]) for f in flds]) + 1

            match self.benchmark_descriptor:
                case BenchmarkModelPath(_, model_path, _, _):
                    output_postfix = model_path.parts[-1]
                case BenchmarkTrajFolder(folder, _):
                    output_postfix = folder.parts[-1]
                case BenchmarkOldDir(folder, _):
                    output_postfix = "RERUN_" + folder.parts[-1]
                case BenchmarkWestpaDir(folder):
                    output_postfix = "WESTPA_" + folder.parts[-1]

            self.output_dir = Path(self.machine.sims_store_dir).joinpath('%06d' % simNr + '_' + output_postfix)


        self.log_dir = self.output_dir

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving simulations to {self.log_dir}")

    def benchmark_protein(self, protein_name: str) -> dict:
        model_output_path: Path | None = None
        match self.benchmark_descriptor:
            case BenchmarkModelPath(_, model_folder, _, _, num_steps, num_save_steps, trajs_per_protein):
                prior_params = json.load(open(os.path.join(model_folder, "prior_params.json"), "r"))
                random_starting_poses: list[NativeTrajPath] = random.choices(self.starting_poses[protein_name], k=trajs_per_protein)
                logging.debug(f"random_starting_poses: {random_starting_poses} {random_starting_poses[0].pdb_top_path}")
            
                if not self.only_gen_cache:
                    assert generate_trajs_semaphore is not None
                    assert find_gpu_mutex is not None
                    assert gpu_list is not None
                    assert available_gpus is not None
                    with generate_trajs_semaphore:
                        with find_gpu_mutex:
                            gpu_idx = None
                            for i in range(len(gpu_list)):
                                if available_gpus[i]:
                                    available_gpus[i] = False
                                    gpu_idx = i
                                    break
                            assert gpu_idx is not None

                    logging.info(f"starting to run model on protein {protein_name}")
                    # run 10 replicas of the model with simulate.py - also CGed. function returns strings pointing to h5 files 
                    model_output_path = self.run_model(random_starting_poses,
                                                       gpu_list[gpu_idx],
                                                       protein_name,
                                                       num_steps,
                                                       num_save_steps)
                    available_gpus[gpu_idx] = True
                    logging.info(f"finished running model {protein_name}")
            case _:
                
                prior_params = {"prior_configuration_name":"CA_Majewski2022_v1"}

        

        with load_trajs_semaphore:
            native_paths = self.native_paths[protein_name]
            logging.info(f"start loading tica model {protein_name}")
            component_analysis_model: DimensionalityReduction
            component_analysis_cache_filename: str
            match self.component_analysis:
                case ComponentAnalysisTypes.TICA:
                    component_analysis_cache_filename = os.path.join(self.machine.cache_path, f"{protein_name}_{self.temperature}K_stride{NATIVE_PATHS_STRIDE}.tica")
                    component_analysis_model = load_cache_or_make_new(
                        Path(component_analysis_cache_filename),
                        lambda: generate_tica_model_from_scratch(
                            native_paths,
                            prior_params,
                            NATIVE_PATHS_STRIDE),
                        TicaModel,
                        self.force_cache_regen
                    )
                case ComponentAnalysisTypes.PCA:
                    component_analysis_cache_filename = os.path.join(self.machine.cache_path, f"{protein_name}_{self.temperature}K.pca")
                    component_analysis_model = load_cache_or_make_new(
                        Path(component_analysis_cache_filename),
                        lambda: generate_pca_model_from_scratch(
                            native_paths,
                            prior_params,
                            NATIVE_PATHS_STRIDE),
                        PCAModel,
                        self.force_cache_regen
                    )

            component_analysis_model_filename = component_analysis_cache_filename
            logging.info(f"finished loading tica model {protein_name}")

            logging.info(f"start loading model trajs {protein_name}")

            stationary_filename: str| None = None
            
            match self.benchmark_descriptor:
                case BenchmarkModelPath(_, _, _, _):
                    assert model_output_path is not None
                    model_trajs: list[ModelTraj] = load_model_traj_pickle(model_output_path)
                    gen_pickle_path = model_output_path
                case BenchmarkTrajFolder(_, traj_paths):
                    model_trajs: list[ModelTraj] = [load_model_traj(path) for path in traj_paths]
                    # save as pickle files for genReport to work
                    gen_pickle_path = f"{os.path.join(self.output_dir, protein_name)}_model_replicas.pkl"
                    with open(gen_pickle_path, "wb") as f:
                         pickle.dump(dict(mdtraj_list=[x.trajectory for x in model_trajs], topology=None, title=""), f)
                case BenchmarkOldDir(_, proteins_paths):
                    model_trajs: list[ModelTraj] = load_model_traj_pickle(proteins_paths[protein_name])
                    gen_pickle_path = f"{os.path.join(self.output_dir, protein_name)}_model_replicas.pkl"
                    # duplicate pickle files for genReport to work
                    with open(gen_pickle_path, "wb") as f:
                         pickle.dump(dict(mdtraj_list=[x.trajectory for x in model_trajs], topology=None, title=""), f)
                case BenchmarkWestpaDir(westpa_dir):
                    sim_config = extract_simulation_config(cfg_path=os.path.join(westpa_dir, "west.cfg"))
                    aa_topology = mdtraj.load(sim_config['topology_path'])
                    prior_name = prior_params["prior_configuration_name"]
                    prior_builder = preprocess.prior_types[prior_name]()
                    atoms_idx = prior_builder.select_atoms(aa_topology.topology)
                    
                    topology = apply_cg(aa_topology, atoms_idx).topology


                    traj_dirs = sorted(glob.glob(os.path.join(westpa_dir, "traj_segs/*/*")))
                    trajectories = []
                    for td in traj_dirs:
                        try:
                            seg_npz_data = numpy.load(os.path.join(td, "seg.npz"))
                        except Exception as e:
                            # Don't error out if we hit an unfinished segment
                            logging.warning(f"Error appending {td}")
                            logging.warning(f"{e}")
                            continue
                        traj = mdtraj.Trajectory(xyz=seg_npz_data["pos"]*0.1, topology=topology)
                        trajectories.append(traj)

                    components = [0]
                    component_values = []
                    for traj in trajectories:
                        values = calculate_component_values(component_analysis_model, traj, components)
                        component_series = numpy.array(values[components[0]])  # Only the first component
                        component_values.append(component_series)

                    num_bins = 80
                    component_min = min([min(cv) for cv in component_values])
                    component_max = max([max(cv) for cv in component_values])
                    #component_bins = numpy.linspace(component_min, component_max, num_bins + 1)

                    binned_components = [
                        numpy.clip((num_bins * (cv - component_min) / (component_max - component_min)).astype(int), None, num_bins - 1)
                        for cv in component_values
                    ]

                    transition_matrix = numpy.zeros((num_bins, num_bins), dtype=int)
                    for binned_traj in binned_components:
                        for j in range(len(binned_traj) - 1):
                            transition_matrix[binned_traj[j], binned_traj[j + 1]] += 1
                    transition_prob_matrix = transition_matrix.astype(numpy.double)/numpy.sum(transition_matrix, axis=1, keepdims=True)
                    transition_prob_matrix = numpy.nan_to_num(transition_prob_matrix, nan=0.0)


                    eigenvalues, eigenvectors = numpy.linalg.eig(transition_prob_matrix.T)
                    stationary_vector = eigenvectors[:, numpy.isclose(eigenvalues, 1)]
                    assert stationary_vector.shape[1] == 1, "Stationary distribution is not unique"
                    stationary_distribution = stationary_vector / numpy.sum(stationary_vector)
                    stationary_distribution = stationary_distribution.real.flatten()
                    stationary_filename = f"{os.path.join(self.output_dir, protein_name)}_stationary.npy"
                    with open(stationary_filename, 'wb') as f:
                        output = numpy.vstack(
                            [numpy.linspace(component_min, component_max, num_bins),
                             stationary_distribution])
                        numpy.save(f, output)
                    
                    gen_pickle_path = f"{os.path.join(self.output_dir, protein_name)}_model_replicas.pkl"
                    with open(gen_pickle_path, "wb") as f:
                        pickle.dump(dict(mdtraj_list=trajectories, topology=None, title=""), f)



            logging.info(f"finished loading model trajs {protein_name}")

            logging.info(f"started loading native trajs {protein_name}")
            native_trajs, all_native_file_strided = load_native_trajs_stride(native_paths, prior_params, NATIVE_PATHS_STRIDE, self.machine.cache_path, protein_name, self.force_cache_regen, self.temperature)
            logging.info(f"finished loading native trajs {protein_name}")

            msm_model_cache_path: str | None = None
            if self.make_table:
                msm_model_cache_path = os.path.join(self.machine.cache_path, f"MSM_native_trajs_{protein_name}_{self.temperature}K.pkl")

                msm_model: MsmRmsdStatistics = load_cache_or_make_new(
                    Path(msm_model_cache_path),
                    lambda: do_msm_analysis(
                        protein_name,
                        [t.trajectory for t in native_trajs],
                        component_analysis_model,
                        prior_params,
                        self.machine.rmsd_dir),
                    MsmRmsdStatistics,
                    self.force_cache_regen
                )

                del msm_model
                    
                        
            logging.info(f"started making native contact map for {protein_name}")
            contact_map_filename, _, = get_contact_maps([x.trajectory for x in native_trajs], protein_name, self.output_dir, self.force_cache_regen, temperature=self.temperature)
            logging.info(f"finished making native contact map for {protein_name}")
            

            logging.info(f"started making reaction coordinates for {protein_name}")
            reaction_coord_kde_filename, _ = get_reaction_coordinate_kde([x.trajectory for x in native_trajs], protein_name, self.machine.cache_path, self.force_cache_regen, self.temperature)
            logging.info(f"finished making reaction coordinates for {protein_name}")

            logging.info(f"started making bond angles for {protein_name}")
            bond_angles_filename, _, _, _ = get_bond_angles_cached(native_trajs, protein_name, self.output_dir, self.force_cache_regen, temperature=self.temperature)
            logging.info(f"finished making bond angles for {protein_name}")

        benchmark_output = {
            "gen_pickle_path": gen_pickle_path,
            "stationary_filename": stationary_filename,
            "tica_model": component_analysis_model_filename,
            "contact_map": contact_map_filename,
            "reaction_coord_kde": reaction_coord_kde_filename,
            "bond_angles_filename": bond_angles_filename,
            "native_paths": [x.__dict__ for x in native_paths],
            "all_native_file_strided": all_native_file_strided,
            "args": sys.argv,
            "benchmark_descriptor": asdict(self.benchmark_descriptor),
            "msm_model": msm_model_cache_path
        }
        
        
   
        del native_trajs #force gc to clean up memory earlier

        logging.info(f"finished benchmarking protein {protein_name}")
        return benchmark_output


    def run_model(self,
                  starting_points: list[NativeTrajPath],
                  gpu: int,
                  protein_name: str,
                  num_steps: int,
                  save_steps: int
                  ) -> Path:
        """
        run a simulation on the gpu id at the starting point

        returns a path to the pickle file outputted by by simulate.py
        """
        assert isinstance(self.benchmark_descriptor, BenchmarkModelPath)
        
        traj_path = self.output_dir.joinpath(f"{protein_name}_model_replicas.pkl")
        
        new_envs = os.environ.copy()
        new_envs["CUDA_VISIBLE_DEVICES"] = f"{gpu}"

        logging.debug('===== Will run simulations for protein %s on GPU %d ======' % (protein_name, gpu))

        # need to pass gpu number, the env variable cannot be used anymore
        # prepSim(model_path, [x.pdb_top_path for x in starting_points], temperature=temperature, output=traj_path, steps=1000, save_steps=10, verbose=False, gpu=gpu)
        
        if self.benchmark_descriptor.checkpoint_path is not None:
            modelToRun = self.benchmark_descriptor.checkpoint_path
        else:
            modelToRun = self.benchmark_descriptor.model_folder

        with open(os.path.join(self.log_dir, f"{protein_name}.log"), "w") as outfile:
            cmdList = [
                    "./simulate.py",
                    modelToRun
                ] + [x.pdb_top_path for x in starting_points] + [
                    "--temperature", f"{self.temperature}",
                    # 100,000 steps with 20 random starting points should be enough for a good coverage of the TICA space, I tested it. it is very slightly biased towards the starting points, but not by much.
                    "--steps", "%d" % num_steps,  # 100,000 steps should take around 12-15 min on 4 GPUs
                    "--save-steps", "%d" % save_steps,
                    "--output", traj_path,
                ]

            if self.benchmark_descriptor.prior_only:
                cmdList.append("--prior-only")

            if self.benchmark_descriptor.prior_nn is not None:
                cmdList.extend(["--prior-nn", str(self.benchmark_descriptor.prior_nn)])

            logging.debug(f"running command \"{' '.join(cmdList)}\"")
            subprocess.run(
                cmdList,
                check=True,
                env=new_envs,
                stdout=outfile,
                stderr=outfile)

        return traj_path
    
    def runParallel(self) -> Path:
        with ThreadPool(num_threads) as pool:
            logging.info("Launching ThreadPool")
            results = pool.map(self.benchmark_protein, self.proteins)

        # benchmarks = self.buildDict(results)
        benchmarks = {
            protein: result
            for protein, result in zip(self.proteins, results)
        }

        benchmarkFile = self.output_dir.joinpath("benchmark.json")
        with open(benchmarkFile, "w") as f:
            match self.benchmark_descriptor:
                case BenchmarkModelPath(_, model_folder, _, _):
                    model_path = model_folder
                case _:
                    model_path = None

            f.write(json.dumps(dict_str_paths({
                "proteins": benchmarks,
                "temperature": self.temperature,
                "used_cache": not self.force_cache_regen,
                "model_path": model_path,
                "rmsd_dir": self.machine.rmsd_dir
            }), indent=4))

        return benchmarkFile
        

class Benchmark350(Benchmark):
    def __init__(
            self,
            to_benchmark: Benchmarkables,
            use_cache: bool,
            machine_c: MachineConf,
            proteins: list[str],
            output_dir_c: Path | None,
            only_gen_cache: bool,
            component_analysis: ComponentAnalysisTypes,
            make_table: bool
    ) -> None:
        self.temperature = 350
        super().__init__(to_benchmark, use_cache, machine_c, proteins, output_dir_c, only_gen_cache, component_analysis, make_table)
    
        self.native_paths = {}
        self.starting_poses = {}
        for p in self.proteins:
            path = os.path.join(self.machine.data_350_path, f"{p}_ca_coords.npy")
            self.native_paths[p] = [NativeTrajPathNumpy(path, get_top_path(path))]
            self.starting_poses[p] = get_native_paths(os.path.join(self.machine.data_300_path, p), self.force_cache_regen)#todo: 350K uses 300K data for random starting poses



class Benchmark300(Benchmark):
    def __init__(
            self,
            to_benchmark: Benchmarkables,
            use_cache: bool,
            machine_c: MachineConf,
            proteins: list[str],
            output_dir_c: Path | None,
            only_gen_cache: bool,
            component_analysis: ComponentAnalysisTypes,
            make_table: bool
    ) -> None:
        self.temperature = 300
        super().__init__(to_benchmark, use_cache, machine_c, proteins, output_dir_c, only_gen_cache, component_analysis, make_table)
        self.native_paths = {}
        for p in proteins:
            self.native_paths[p] = get_native_paths(os.path.join(self.machine.data_300_path, p), self.force_cache_regen)
        self.starting_poses = self.native_paths



def did_path_finish_simulating(path: str) -> bool:
    finished_path = os.path.join(path, "simulation", "finished.txt")
    if os.path.isfile(finished_path):
        with open(finished_path) as finished_file:
            had_error = 'error' in finished_file.read()
            return not had_error
    return False

def get_native_paths(folder: str, force_cache_regen: bool) -> list[NativeTrajPath]:
    def make_path(base: str):
        basename = os.path.basename(base)
        h5_path = os.path.join(base, "result", f"output_{basename}.h5")
        pdb_path = os.path.join(base, "processed", f"{basename}_processed.pdb")
        return NativeTrajPathH5(h5_path, pdb_path)

    f = os.path.join(folder, "native_paths.pkl")

    def load_native_paths() -> list[NativeTrajPath]:
        return [make_path(x) for x in
            sorted(list(filter(did_path_finish_simulating, glob.glob(os.path.join(folder, "*")))))]

    return load_cache_or_make_new(
        Path(f),
        load_native_paths,
        list,
        force_cache_regen)

def get_top_path(coord_path: str) -> str:
    dir_path = os.path.dirname(coord_path[:-len("_coords.npy")])
    base = os.path.basename(coord_path[:-len("_coords.npy")]) + ".pdb"
    out = os.path.join(dir_path, "topology", base)
    return out


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-path", default=None, help="The path of the model to benchmark")
    arg_parser.add_argument("--temperature", type=int, help="Temperature in Kelvin of the model")
    arg_parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True, help="Regenerate the cache instead of using previous runs data")
    arg_parser.add_argument("--only-gen-cache", action=argparse.BooleanOptionalAction, default=False, help="Only regenerate the stuff that is cached like the TICA model, will not run the model at all")
    arg_parser.add_argument("--machine", type=str, default=None, choices=machines.keys(), help="Which server this is being run on")
    arg_parser.add_argument("--proteins", type=str, default=None, help="Proteins to run benchmark on", nargs="+")
    arg_parser.add_argument("--output-dir", type=Path, default=None, help="Output directory of benchmarks")
    arg_parser.add_argument("--prior-only", default=False, action='store_true', help="Disable the model and use only the prior forcefield")
    arg_parser.add_argument("--prior-nn", default=None, type=Path, help="Path to the folder of a neural network prior.")
    arg_parser.add_argument("--gpus", default=None, type=str, help="List of GPUs to use (e.g. \"0,1,2\")")
    arg_parser.add_argument("--disable-wandb", action=argparse.BooleanOptionalAction, default=False, help="Disable wandb logging")
    arg_parser.add_argument("--trajs-folder", type=Path, default=None, help="Directory containing the trajectories of the proteins")
    arg_parser.add_argument("--num-steps", type=int, default=100000, help="Number of steps to simulate for trajectory")
    arg_parser.add_argument("--num-save-steps", type=int, default=1000, help="Save point on trajectory every N steps")
    arg_parser.add_argument("--trajs-per-protein", type=int, default=20, help="How many replicas to simulate for each protein")
    arg_parser.add_argument("--component-analysis-type", type=str, default="TICA", choices=["TICA", "PCA"], help="Which type of dimensionality reduction to use")
    arg_parser.add_argument("--old-benchmark-dir", type=Path, default=None, help="Old benchmark directory to re-run on")
    arg_parser.add_argument("--westpa-dir", type=Path, default=None, help="Westpa output to benchmark")
    arg_parser.add_argument("--calc-kl-divergence", action=argparse.BooleanOptionalAction, default=False,
                            help="Calculate the KL divergence for components in tica space")
    arg_parser.add_argument("--enable-msm-metrics", action=argparse.BooleanOptionalAction, 
                       default=True, help="Disable generation of native macrostate statistics")
    
    args = arg_parser.parse_args()

    assert ((args.model_path is not None) ^
            (args.trajs_folder is not None) ^
            (args.old_benchmark_dir is not None) ^
            (args.westpa_dir is not None)), "Must have exactly one of model, trajectory, or old benchmark"
    

    if args.gpus:
        gpu_ids = [int(i) for i in args.gpus.strip().split(",")]
    else:
        gpu_ids = [*range(torch.cuda.device_count())]
    init_gpu_locks(gpu_ids)

    if args.model_path is None:
        assert args.prior_only == False, "invalid option"
        assert args.prior_nn is None, "invalid option"

    run_individual_plots = True
    if args.model_path is not None:
        to_benchmark = ModelPath(
            args.model_path,
            args.prior_only,
            args.prior_nn,
            args.num_steps,
            args.num_save_steps,
            args.trajs_per_protein)
    elif args.trajs_folder is not None:
        args.disable_wandb = True
        to_benchmark = TrajFolder(args.trajs_folder)
    elif args.old_benchmark_dir is not None:
        to_benchmark = OldBenchmarkRerun(args.old_benchmark_dir)
    elif args.westpa_dir is not None:
        to_benchmark = WestpaFolder(args.westpa_dir)
        run_individual_plots = False
    else:
        assert False

    component_analysis_type: ComponentAnalysisTypes
    match args.component_analysis_type:
        case "TICA":
            component_analysis_type = ComponentAnalysisTypes.TICA
        case "PCA":
            component_analysis_type = ComponentAnalysisTypes.PCA
        case _:
            assert False
        
    if args.machine is not None:
        assert args.machine in machines, f"Unknown machine name, must be one of: {', '.join(machines.keys())}"
        machine_conf = machines[args.machine]
    else:
        script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        machine_config_filename = os.path.join(script_dir, "gen_benchmark.conf")
        assert os.path.exists(machine_config_filename), f"--machine was not given and no local config was found: {machine_config_filename}"
        with open(machine_config_filename, 'r') as file:
            machine_config_data = yaml.safe_load(file)
            machine_conf = MachineConf(**machine_config_data)

    # put the code below into a separate function
    if args.temperature == 350:
        logging.info('Running at 350K')
        benchmark = Benchmark350(to_benchmark, args.use_cache, machine_conf, args.proteins, args.output_dir, args.only_gen_cache, component_analysis_type, args.enable_msm_metrics)
    elif args.temperature == 300:
        logging.info('Running at 300K')
        benchmark = Benchmark300(to_benchmark, args.use_cache, machine_conf, args.proteins, args.output_dir, args.only_gen_cache, component_analysis_type, args.enable_msm_metrics)
    else:
        assert False, "temperature must be either 300 or 350"


    benchmarkFile = benchmark.runParallel()
    if args.only_gen_cache:
        return
    runReport(benchmarkFile,
              also_plot_locally=True,
              do_rmsd_metrics=args.enable_msm_metrics,
              do_kl_divergence=args.calc_kl_divergence,
              disable_wandb=args.disable_wandb,
              plot_individuals=run_individual_plots)


def dict_str_paths(d: dict) -> dict:
    keys = d.keys()
    for k in keys:
        if isinstance(d[k], dict):
            d[k] = dict_str_paths(d[k])
        elif isinstance(d[k], Path):
          d[k] = str(d[k])
    return d
            

if __name__ == "__main__":
    main()
