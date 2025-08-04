#!/usr/bin/env python3
import argparse
import matplotlib
import matplotlib.cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
import json
import wandb
import numpy
import numpy.typing
import numpy as np
import pickle
from report_generator.traj_loading import load_model_traj_pickle, ModelTraj
from report_generator.tica_plots import calc_atom_distance, DimensionalityReduction
from report_generator.reaction_coordinate import calc_reaction_coordinate, ReactionCoordKde
from report_generator.contact_maps import make_contact_map_plot, make_contact_map, ContactMap
from report_generator.bond_and_angle_analysis import plot_bond_length_angles, get_bond_angles
from report_generator.kullback_leibler_divergence import kl_div_calc, wasserstein, wasserstein1d
import matplotlib.patches as mpatches
import matplotlib.colorizer
from report_generator.msm_analysis import do_msm_analysis, MsmRmsdStatistics, get_expiremental_structure
import matplotlib.colors as colors
import scipy
import time
import mdtraj
from tabulate import tabulate
from wandb.errors import CommError
import logging

def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("benchmark", type=Path, help="The path to benchmak json")
    arg_parser.add_argument("--test-run", action=argparse.BooleanOptionalAction, default=False, help="Run faster but not on all of the data")
    arg_parser.add_argument("--also-plot-locally", action=argparse.BooleanOptionalAction, default=False, help="In addition to uploading to wandb, also plot everything to a local folder")
    args = arg_parser.parse_args()

    runReport(args.benchmark, args.also_plot_locally, False, True, False, False)

def runReport(
        benchmark: Path,
        also_plot_locally: bool,
        do_rmsd_metrics: bool,
        do_kl_divergence: bool,
        disable_wandb: bool,
        plot_individuals: bool
) -> None:
    with open(benchmark) as f:
        benchmark_name = benchmark.parts[-2]
        benchmark_data = json.loads(f.read())
        benchmarkFolder = benchmark.parent
        trajPlotFolder = Path(benchmarkFolder).joinpath("traj_plots")
        Path(trajPlotFolder).mkdir(parents=True, exist_ok=True)

        if (not disable_wandb):
            wandb.init(project="andy_report", config={"json_input": benchmark_data,
                                                      "used_cache": benchmark_data["used_cache"],
                                                      "model_path": benchmark_data["model_path"]})
        
        metrics_dicts = {}
        for protein_name, info in benchmark_data["proteins"].items():
            logging.info(f'Processing protein: {protein_name}')
            protein_name: str
            model_trajs_raw: list[ModelTraj] = load_model_traj_pickle(info["gen_pickle_path"])
            total_before_nan_filter = len(model_trajs_raw)
            model_trajs = [x[1] for x in filter(lambda x: check_if_traj_has_nans(protein_name, x[0], x[1]), enumerate(model_trajs_raw))]
            logging.info(f"trajectories of {protein_name} without nan is {len(model_trajs)}/{total_before_nan_filter}")
            #Take out the first 10%% of frames in model_trajs in order not to bias the model towards the starting points
            # take out the first 10% of frames in model_trajs as they are biased towards the starting points and thus towards the GT/native data (remember starting points are sampled from GT data)
            model_trajs = [t.filterFrames(start=int(t.trajectory.n_frames/10)-1, end=-1) for t in model_trajs]
            with open(info["tica_model"], 'rb') as tica_model_file:
                with open(info["reaction_coord_kde"], 'rb') as reaction_coord_kde:
                    with open(info["contact_map"], 'rb') as contact_map:
                        with open(info["bond_angles_filename"], 'rb') as bond_angles_file:
                            with open(info["all_native_file_strided"], 'rb') as all_native_file_strided:
                                tica_model = pickle.load(tica_model_file)
                                native_rc_kde = pickle.load(reaction_coord_kde)
                                contact_map_native = pickle.load(contact_map)
                                dict = pickle.load(bond_angles_file)
                                bond_lengths_native = dict['bond_lengths']
                                bond_angles_native = dict['bond_angles']
                                dihedrals_native = dict['dihedrals']
                                
                                native_trajs_strided = pickle.load(all_native_file_strided)
                                
                                assert isinstance(tica_model, DimensionalityReduction)
                                # assert all([isinstance(x, scipy.stats.gaussian_kde) for x in native_kde])
                                assert isinstance(native_rc_kde, ReactionCoordKde)
                                assert isinstance(contact_map_native, ContactMap)


                                if plot_individuals:
                                    for i, raw_tica_coords in enumerate(tica_model.decompose([calc_atom_distance(x.trajectory) for x in model_trajs_raw])):
                                        traj = model_trajs_raw[i]
                                        assert isinstance(traj.trajectory.xyz, np.ndarray)
                                        did_explode = np.isnan(traj.trajectory.xyz).any()

                                        filename = f"{protein_name}_{i}_EXPLODED.png" if did_explode else f"{protein_name}_{i}.png"
                                    
                                        plot_trajectory_raw(raw_tica_coords, trajPlotFolder.joinpath(filename))
                                
                                prior_params = {"prior_configuration_name":"CA_Majewski2022_v1"} #TODO don't hard code this

                                msm_model: MsmRmsdStatistics | None = None

                                if do_rmsd_metrics:
                                    msm_model = do_msm_analysis(
                                        protein_name,
                                        [t.trajectory for t in model_trajs], 
                                        tica_model,
                                        prior_params,
                                        Path(benchmark_data["rmsd_dir"]))

                                exp_structure = get_expiremental_structure(Path(benchmark_data["rmsd_dir"]), protein_name, prior_params)
                                native_msm_model: MsmRmsdStatistics | None = None

                                if info["msm_model"] is not None:
                                    with open(info["msm_model"], "rb") as f:
                                        native_msm_model = pickle.load(f)
                                        assert isinstance(native_msm_model, MsmRmsdStatistics)
                                if msm_model is not None:
                                    if native_msm_model is not None:
                                        with open(benchmarkFolder.joinpath(f"msm_{protein_name}.json"), "w") as f:
                                            f.write(json.dumps({
                                                "native_mean": native_msm_model.native_rmsd_mean,
                                                "native_stddev": native_msm_model.native_rmsd_stddev,
                                                "native_macro_prob": native_msm_model.native_macro_prob,
                                                "model_mean": msm_model.native_rmsd_mean,
                                                "model_stddev": msm_model.native_rmsd_stddev,
                                                "model_macro_prob": msm_model.native_macro_prob,
                                        }, indent=4))


                                stationary_distribution: None | numpy.typing.NDArray = None
                                if info["stationary_filename"] is not None:
                                    with open(info["stationary_filename"], 'rb') as f:
                                        stationary_distribution = numpy.load(f)
                                            
                                fig_tica_spaces, fig_contours, fig_contact_map, fig_pdfs, fig_gyration_pdf, metrics_dict = make_figs(
                                    benchmark_name,
                                    protein_name,
                                    model_trajs,
                                    native_trajs_strided,
                                    tica_model,
                                    native_rc_kde,
                                    contact_map_native,
                                    bond_lengths_native,
                                    bond_angles_native,
                                    dihedrals_native,
                                    native_msm_model,
                                    msm_model,
                                    exp_structure,
                                    do_kl_divergence,
                                    stationary_distribution
                                )

                                metrics_dicts[protein_name] = metrics_dict
                                


                                if (not disable_wandb):
                                    wandb.log({
                                        f"plots_tica_space_{protein_name}": wandb.Image(fig_tica_spaces),
                                        f"plots_contour_{protein_name}": wandb.Image(fig_contours),
                                        f"plots_contact_map_{protein_name}": wandb.Image(fig_contact_map),
                                        f"plots_pdfs_{protein_name}": wandb.Image(fig_pdfs),
                                        f"plots_gyration_pdf_{protein_name}": wandb.Image(fig_gyration_pdf)
                                    })
                                if also_plot_locally:
                                    fig_tica_spaces.savefig(benchmarkFolder.joinpath(Path(f"tica_spaces_{protein_name}.png")))
                                    fig_contours.savefig(benchmarkFolder.joinpath(Path(f"tica_contours_{protein_name}.png")))
                                    fig_contact_map.savefig(benchmarkFolder.joinpath(Path(f"contact_map_{protein_name}.png")))
                                    fig_pdfs.savefig(benchmarkFolder.joinpath(Path(f"plot_pdfs_{protein_name}.png")))
                                    fig_gyration_pdf.savefig(benchmarkFolder.joinpath(Path(f"plot_gyration_pdf_{protein_name}.png")))


        with open(benchmarkFolder.joinpath(f"metrics.json"), "w") as f:
            f.write(json.dumps(metrics_dicts
            , indent=4))
        logging.info(f"metrics = {metrics_dicts}")
        
        kl_tica = {k: x["kls_tica_2d"] for k, x in metrics_dicts.items()}
        kl_rc = {k: x["kls_reaction_coord"] for k, x in metrics_dicts.items()}
        w1_tica = {k: x["wasserstein_tica_2d"] for k, x in metrics_dicts.items()}
        if not None in kl_tica.values():
            kl_tica['all'] = sum(kl_tica.values()) / len(kl_tica)
            kl_rc['all'] = sum(kl_rc.values()) / len(kl_rc)
        else:
            kl_tica['all'] = 0
            kl_rc['all'] = 0

        w1_tica['all'] = sum(w1_tica.values()) / len(w1_tica)

        if benchmark_data["model_path"]:
            # pickle dump all the dictionaries, mostly if ever needed to debug wandb
            with open(benchmark_data["model_path"] + '/kl_tica.pkl', 'wb') as f:
                pickle.dump(kl_tica, f)
            with open(benchmark_data["model_path"] + '/kl_rc.pkl', 'wb') as f:
                pickle.dump(kl_rc, f)
            with open(benchmark_data["model_path"] + '/w1_tica.pkl', 'wb') as f:
                pickle.dump(w1_tica, f)
            
            # pickle load all dictionaries
            with open(benchmark_data["model_path"] + '/kl_tica.pkl', 'rb') as f:
                kl_tica = pickle.load(f)
            with open(benchmark_data["model_path"] + '/kl_rc.pkl', 'rb') as f:
                kl_rc = pickle.load(f)
            with open(benchmark_data["model_path"] + '/w1_tica.pkl', 'rb') as f:
                w1_tica = pickle.load(f)
        
        if (not disable_wandb):
            # create wandb table with all KL results, for each protein and for avg across all proteins

            # Define the artifact name for the table
            artifact_name = "model_metrics"
            table_name = "metrics"
            
            protList = benchmark_data["proteins"].keys()
            columns = ["Model Name", "kl_tica_all", "kl_rc_all", "w1_tica"] + ['kl_tica_' + protein_name for protein_name in protList] + ['kl_rc_' + protein_name for protein_name in protList] + ['w1_tica_' + protein_name for protein_name in protList]
            logging.info(f'columns {columns}')

            modelName: str
            if benchmark_data["model_path"] is not None:
                modelName = Path(benchmark_data["model_path"]).parts[-1]
                if 'checkpoint' in benchmark.parts:
                    modelName += '_ch' + benchmark.parts[-2].split('-')[-1]
            else:
                modelName = "no_model_test123"

            logging.info(f"Saving the model as {modelName}")

            all_proteins_processed = len(protList) == 6
            if all_proteins_processed: 
                regenerateTable = False
                table: wandb.Table
                if not regenerateTable:
                    try:
                        # Try to load the existing artifact
                        artifact = wandb.use_artifact(artifact_name + ':latest')
                        logging.info(f"artifact {artifact}")
                        table_gotten = artifact.get(table_name)
                        assert isinstance(table_gotten, wandb.Table)
                        table = table_gotten
                        logging.info("table {table}")
                        logging.info("Loaded existing table.")
                    except CommError:
                        #Create a new table if the artifact doesn't exist
                        table = wandb.Table(columns=columns)
                        logging.info("Created a new table.")
                else:
                    table = wandb.Table(columns=columns)
                    logging.info("Created a new table.")

                # Add rows to the table for each model
                vals = [kl_tica['all'], kl_rc['all'], w1_tica['all']] + [kl_tica[protein_name] for protein_name in protList] + [kl_rc[protein_name] for protein_name in protList] + [w1_tica[protein_name] for protein_name in protList]
                table.add_data(modelName, *vals)

                # Reinitialize the table with its updated data to ensure compatibility
                table = wandb.Table(columns=columns, data=table.data)

                # Log the updated table to W&B
                wandb.log({table_name: table})

                # Print only the first 3 columns
                num_columns_to_display = 4
                trimmed_columns = columns[:num_columns_to_display]
                trimmed_data = [row[:num_columns_to_display] for row in table.data]

                logging.info("Table contents:")
                logging.info(tabulate(trimmed_data, headers=trimmed_columns, tablefmt="grid"))

                # Save the updated table as an artifact for future runs
                artifact = wandb.Artifact(artifact_name, type="metrics")
                artifact.add(table, table_name)
                wandb.log_artifact(artifact)
            else:
                # don't upload to wandb if not all proteins are processed, only print to the terminal
                
                table = wandb.Table(columns=columns)

                # Add rows to the table for each model
                vals = [kl_tica['all'], kl_rc['all'], w1_tica['all']] + [kl_tica[protein_name] for protein_name in protList] + [kl_rc[protein_name] for protein_name in protList] + [w1_tica[protein_name] for protein_name in protList]
                table.add_data(modelName, *vals)

                    # Print only the first 3 columns
                num_columns_to_display = 4
                trimmed_columns = columns[:num_columns_to_display]
                trimmed_data = [row[:num_columns_to_display] for row in table.data]
                logging.info("Table contents:")
                logging.info(tabulate(trimmed_data, headers=trimmed_columns, tablefmt="grid"))
            # Finish the WandB run
            wandb.finish()



# def make_react_coord_fig(ax, model_trajs: list[ModelTraj], native_rc_kde: ReactionCoordKde) -> Figure:
def make_react_coord(
        ax,
        model_trajs: list[ModelTraj],
        native_rc_kde: ReactionCoordKde,
        do_kl_divergence: bool
) -> float | None:
    model_coords = calc_reaction_coordinate([x.trajectory for x in model_trajs])
    
    stride = 1 # do not skip any points, only has 2,000 points
    model_rc_kde = scipy.stats.gaussian_kde(model_coords[::stride])

    xmin = min(native_rc_kde.min_val, numpy.min(model_coords))
    xmax = max(native_rc_kde.max_val, numpy.max(model_coords))

    Xs = numpy.linspace(xmin, xmax, num=100)
    ax.set_title('PDF over Reaction Coordinate')
    ax.set_xlim((xmin, xmax))
    ax.set_xlabel("Distance between two chosen atoms")

    stride = 1000 # keep this stride as we have around 1M+ points
    native_dataset = native_rc_kde.model.dataset[:,::stride]
    native_rc_kde_strided = scipy.stats.gaussian_kde(native_dataset)
    
    Ys_model = model_rc_kde(Xs)
    ax.plot(Xs, Ys_model, c="red", label='Model')

    Ys_native = native_rc_kde_strided(Xs)
    ax.plot(Xs, Ys_native, c="blue", label='Ground Truth')
    
    kl_rc: float | None = kl_div_calc(model_rc_kde, native_rc_kde_strided, native_dataset) if do_kl_divergence else None

    return kl_rc

def make_figs(
        benchmark_name: str,
        protein_name: str,
        model_trajs: list[ModelTraj],
        native_trajs_strided: list[ModelTraj],
        tica_model: DimensionalityReduction,
        native_rc_kde: scipy.stats.gaussian_kde,
        contact_map_native: ContactMap,
        bond_lengths_native,
        bond_angles_native,
        dihedrals_native,
        native_macrostates_positions: MsmRmsdStatistics | None,
        model_msm_model: MsmRmsdStatistics | None,
        exp_structure: mdtraj.Trajectory,
        do_kl_divergence: bool,
        stationary_distribution: None | numpy.typing.NDArray
) -> tuple[Figure, Figure, Figure, Figure, Figure, dict[str, list[float] | float | None]]:
    NUM_TICA_PLOTS = 3
    fig_tica_spaces, axes_tica_spaces = plt.subplots(nrows=NUM_TICA_PLOTS, ncols=3, squeeze=False, figsize=(15, 15))
    fig_contours, axes_contours = plt.subplots(nrows=1, ncols=1, squeeze=False)
    fig_contact_map, axes_contact_map = plt.subplots(nrows=1, ncols=1, squeeze=False)
    fig_pdfs, axes_pdfs = plt.subplots(nrows=2, ncols=4, squeeze=False, figsize=(15, 10))
    fig_gyration_pdfs, axes_gyration_pdfs = plt.subplots(nrows=1, ncols=1, squeeze=False)
    
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # assert isinstance(axes, numpy.ndarray)

    fig_tica_spaces.suptitle(protein_name)
    fig_contours.suptitle(protein_name)
    fig_contact_map.suptitle(protein_name)
    fig_pdfs.suptitle(protein_name)
    fig_gyration_pdfs.suptitle(protein_name)
    # bname = benchmark_name
    # if len(bname) > 100:
    #     bname = bname[:50] + "..." + bname[-50:]
    # fig.text(0.5, 0.94, bname, va="top", ha="center", size=10)

    model_tica_datas: list[numpy.typing.NDArray] = tica_model.decompose([calc_atom_distance(x.trajectory) for x in model_trajs])
    model_projected_data = numpy.concatenate(model_tica_datas)


    native_proj_datas: list[numpy.typing.NDArray]  = tica_model.decompose([calc_atom_distance(x.trajectory) for x in native_trajs_strided])
    assert not np.isnan(native_proj_datas).any()
    
    strideNative = 10
    native_proj_datas_concat = numpy.concatenate(native_proj_datas)[::strideNative]
    assert not np.isnan(native_proj_datas_concat).any()
    
    def get_boundary(tic: int)-> tuple[float, float]:
        xmin = np.min([numpy.min(model_projected_data[:, tic]), np.min(native_proj_datas_concat[:, tic])])
        xmax = np.max([numpy.max(model_projected_data[:, tic]), np.max(native_proj_datas_concat[:, tic])])
        x_bot = xmin - 0.1 * (xmax-xmin)
        x_top = xmax + 0.1 * (xmax-xmin)

        assert not np.isnan(x_bot)
        assert not np.isnan(x_top)
        return x_bot, x_top
        
        
    for i in range(NUM_TICA_PLOTS):
        x_bot, x_top = get_boundary(i)
        y_bot, y_top = get_boundary(i+1)
        if i == 0:
            axes_contours[0, 0].set_xlim((x_bot, x_top))
            axes_contours[0, 0].set_ylim((y_bot, y_top))
            axes_contours[0, 0].set_xlabel(f"{tica_model.get_axis_name()}{i}")
            axes_contours[0, 0].set_ylabel(f"{tica_model.get_axis_name()}{i+1}")
        for j in range(0,3):
            axes_tica_spaces[i, j].set_xlim((x_bot, x_top))
            axes_tica_spaces[i, j].set_ylim((y_bot, y_top))
            axes_tica_spaces[i, j].set_xlabel(f"{tica_model.get_axis_name()}{i}")
            axes_tica_spaces[i, j].set_ylabel(f"{tica_model.get_axis_name()}{i+1}")
        
    start_time = time.time()
    x_bot, x_top = get_boundary(0)
    y_bot, y_top = get_boundary(1)
    X, Y = numpy.mgrid[x_bot:x_top:50j, y_bot:y_top:50j]

    positions = numpy.vstack([X.ravel(), Y.ravel()])
    strideModelTica2D = 1

    model_kde_2d = scipy.stats.gaussian_kde(model_projected_data[::strideModelTica2D, :2].transpose())
    native_kde_2d = scipy.stats.gaussian_kde(native_proj_datas_concat[:, :2].transpose())
    
    Z_native_1D = native_kde_2d(positions)
    Z_native = numpy.reshape(Z_native_1D.T, X.shape)

    Z_model_1D = model_kde_2d(positions)
    Z_model = numpy.reshape(Z_model_1D.T, X.shape)
    logging.info("done using model 2d kde")


    levels = 7
    axes_contours[0, 0].set_title("Ground truth PDF: Model vs GT")
    cont_gt = axes_contours[0, 0].contour(X, Y, Z_native, cmap=matplotlib.colormaps["Blues"], linewidths=2, levels=levels)
    cont_model = axes_contours[0, 0].contour(X, Y, Z_model, cmap=matplotlib.colormaps["Reds"], linewidths=2, levels=levels)
    axes_contours[0, 0].legend([cont_gt.legend_elements()[0][-1], cont_model.legend_elements()[0][-1]], ["Ground Truth", "Model"])

    kl_tica_2d = kl_div_calc(model_kde_2d, native_kde_2d, native_proj_datas_concat[:, :2].T) if do_kl_divergence else None
    
    w1_tica2D = wasserstein(native_kde_2d, model_kde_2d, x_bot, x_top, y_bot, y_top)
    
    def plot_points_with_colors(
            ax: Axes,
            projected_data: list[numpy.typing.NDArray],
            msm_info: MsmRmsdStatistics | None,
            tic_x: int,
            tic_y: int,
            tic_z: int, #backup in case MSM model is missing
            stride: int
    ):
        exp_structure_projected = tica_model.decompose([calc_atom_distance(exp_structure)])[0]
        for datas in projected_data:
            crystal_structures = ax.scatter(exp_structure_projected[:, tic_x], exp_structure_projected[:, tic_y], c="black", s=30, marker="^", zorder=100, label="exp. structure")
            if msm_info is not None:
                microstate_assignemnt = msm_info.microstate_kmeans.transform(datas[::stride, :msm_info.num_tica_components_used])
                macrostate_assignments_maybe: list[int | None] = [msm_info.macrostate_assigments[x] for x in microstate_assignemnt]
                
                macrostate_colors: list[int] = [0 if x is None else x+1 for x in macrostate_assignments_maybe]
                #https://stackoverflow.com/questions/36377638/how-to-map-integers-to-colors-in-matplotlib
                cmap = colors.ListedColormap(["black", "blue", "green", "red", "magenta", "yellow", "cyan"])
                
                norm = colors.BoundaryNorm(list(range(cmap.N)), cmap.N)
                colorizer = matplotlib.colorizer.Colorizer(cmap= cmap, norm=norm)
                ax.scatter(datas[::stride, tic_x], datas[::stride, tic_y], c=macrostate_colors, colorizer=colorizer, s=1, alpha=1)
                native_color_patch = mpatches.Patch(color=colorizer.to_rgba(numpy.array([msm_info.native_macrostate_id + 1]))[0], label="native macrostate")
                ax.legend(handles=[native_color_patch, crystal_structures])
            else:
                ax.scatter(datas[::stride, tic_x], datas[::stride, tic_y], c=datas[::stride, tic_z], s=1, alpha=1)
            
            
    
    for tic_level in range(NUM_TICA_PLOTS):
        axes_tica_spaces[tic_level, 0].set_title(f"Model Points in {tic_level}-{tic_level+1} {tica_model.get_title_name()} space")
        axes_tica_spaces[tic_level, 1].set_title(f"Model Points in {tic_level}-{tic_level+1} {tica_model.get_title_name()} space")
        scatter_colors = matplotlib.cm.ScalarMappable(cmap='rainbow').to_rgba(numpy.linspace(0, 1, len(model_tica_datas)))

        nrReplicas = len(model_tica_datas)

        labels: list[str | None] = [f"R{i}" for i in range(nrReplicas)]
        # if there are more than 10 replicas, only show 10 replicas in total in the legend
        if nrReplicas > 10:
            for r in range(nrReplicas):
                if r % int(nrReplicas / 10) != 0:
                    labels[r] = None
        for r, (datas, c) in enumerate(zip(model_tica_datas, scatter_colors)):
            stride = 1
            axes_tica_spaces[tic_level, 0].scatter(datas[::stride, tic_level], datas[::stride, tic_level+1], color=c, s=1, alpha=1, label=labels[r])
        # axes_tica_spaces[tic_level, 0].legend(loc='center', bbox_to_anchor=(1.1, 0.45))

        plot_points_with_colors(
            axes_tica_spaces[tic_level, 1],
            model_tica_datas,
            model_msm_model,
            tic_level,
            tic_level+1,
            tic_level+2,
            1
        )
    

    for tic_level in range(NUM_TICA_PLOTS):
        axes_tica_spaces[tic_level, 2].set_title(f"GT Points in {tica_model.get_title_name()} space")
        plot_points_with_colors(
            axes_tica_spaces[tic_level, 2],
            native_proj_datas,
            native_macrostates_positions,
            tic_level,
            tic_level+1,
            tic_level+2,
            10
        )
        
    kl_rc = make_react_coord(axes_pdfs[0, 0], model_trajs, native_rc_kde, do_kl_divergence)

    contact_map_model = make_contact_map([x.trajectory for x in model_trajs])
    make_contact_map_plot(axes_contact_map[0, 0], contact_map_native, contact_map_model)


    bond_lengths_model, bond_angles_model, dihedrals_model = get_bond_angles(mdtraj.join([x.trajectory for x in model_trajs]))

    label_list = ['Ground truth', 'Model']
    plot_bond_length_angles(axes_pdfs[0, 1], [bond_lengths_native, bond_lengths_model], labels=label_list, title="Bond Length Distribution", xlabel="Length (nm)", colors=['blue', 'red'])
    plot_bond_length_angles(axes_pdfs[0, 2], [bond_angles_native, bond_angles_model], labels=label_list, title="Bond Angle Distribution", xlabel="Angle (Radians)", colors=['blue', 'red'])
    plot_bond_length_angles(axes_pdfs[0, 3], [dihedrals_native, dihedrals_model], labels=label_list, title="Dihedral Distribution", xlabel="Angle (Radians)", colors=['blue', 'red'])
    bond_lengths_native_concat = numpy.concat(bond_lengths_native).flatten()
    bond_lengths_model_concat = numpy.concat(bond_lengths_model).flatten()
    bond_angles_native_concat = numpy.concat(bond_angles_native).flatten()
    bond_angles_model_concat = numpy.concat(bond_angles_model).flatten()
    dihedrals_native_concat = numpy.concat(dihedrals_native).flatten()
    dihedrals_model_concat = numpy.concat(dihedrals_model).flatten()


    logging.info("calculating B-A-D metrics")
    bond_angle_dihedral_kls = []
    bond_angle_dihedral_wasser = []
    for i, (native_data, model_data) in enumerate([
            (bond_lengths_native_concat, bond_lengths_model_concat),
            (bond_angles_native_concat, bond_angles_model_concat),
            (dihedrals_native_concat, dihedrals_model_concat)
    ]):
        logging.info(f"doing {["lengths", "angles", "dihedrals"][i]}")
        native_kde = scipy.stats.gaussian_kde(native_data)
        model_kde = scipy.stats.gaussian_kde(model_data[::30])
        kl = kl_div_calc(native_kde, model_kde, native_data)
        xmin = numpy.min(native_data)
        xmax = numpy.max(native_data)
        wasser = wasserstein1d(native_kde, model_kde, xmin, xmax)
        bond_angle_dihedral_kls.append(kl)
        bond_angle_dihedral_wasser.append(wasser)
    logging.info("done calculating B-A-D metrics")

    
    
    tica_1d_axes = axes_pdfs[1,:]
    nrTICA1Dplots = tica_1d_axes.shape[0]
    subtitles = [f"PDF {tica_model.get_title_name()} component %d" % i for i in range(0, nrTICA1Dplots)]
    wassersteins, kls = make_tica_1ds(
        tica_1d_axes,
        model_projected_data,
        native_proj_datas_concat,
        subtitles,
        tica_model.get_axis_name(),
        stationary_distribution)
    logging.info("--- TICA 1D plots %s seconds ---" % (time.time() - start_time))
    handles, all_labels = axes_pdfs[0, 0].get_legend_handles_labels()
    fig_pdfs.legend(handles, all_labels)



    model_rgs = numpy.concat([mdtraj.compute_rg(x.trajectory) for x in model_trajs])
    native_rgs = numpy.concat([mdtraj.compute_rg(x.trajectory) for x in native_trajs_strided])
    model_kernel_rg = scipy.stats.gaussian_kde(model_rgs)
    native_kernel_rg = scipy.stats.gaussian_kde(native_rgs)
    max_rg = numpy.max(numpy.concat([model_rgs, native_rgs]))
    Xs_rg = numpy.linspace(0, max_rg, num=100)
    Ys_rg_model = model_kernel_rg(Xs_rg)
    Ys_rg_native = native_kernel_rg(Xs_rg)
    axes_gyration_pdfs[0, 0].set_title("Radius of Gyrtaion PDF")
    axes_gyration_pdfs[0, 0].plot(Xs_rg, Ys_rg_model, c="red", label='Model')
    axes_gyration_pdfs[0, 0].plot(Xs_rg, Ys_rg_native, c="blue", label='Ground Truth')
    axes_gyration_pdfs[0, 0].legend()
    kl_gyration = kl_div_calc(native_kernel_rg, model_kernel_rg, native_rgs)
    wasser_gyration = wasserstein1d(native_kernel_rg, model_kernel_rg, 0, max_rg)


    metrics_dics: dict[str, list[float] | float | None] = {
        #reaction coords
        "kls_gyration": kl_gyration,
        "wasser_gyration": wasser_gyration,
        #angles bonds dihedrals
        "bond_kls": bond_angle_dihedral_kls[0],
        "bond_wasser": bond_angle_dihedral_wasser[0],
        "angle_kls": bond_angle_dihedral_kls[1],
        "angle_wasser": bond_angle_dihedral_wasser[1],
        "dihedral_kls": bond_angle_dihedral_kls[2],
        "dihedral_wasser": bond_angle_dihedral_wasser[2],
        #reaction coords
        "kls_reaction_coord": kl_rc,
        #1d ticas
        "wasserstein_tica_1d": wassersteins,
        "kls_tica_1d": kls,
        #2d ticas
        "kls_tica_2d": kl_tica_2d,
        "wasserstein_tica_2d": w1_tica2D,
    }
    
    return fig_tica_spaces, fig_contours, fig_contact_map, fig_pdfs, fig_gyration_pdfs, metrics_dics



def make_tica_1ds(axs: numpy.ndarray,
                  model_projected_data: numpy.typing.NDArray, native_proj_datas_concat: numpy.typing.NDArray, subtitles: list[str],
                  component_names: str,
                  stationary_distribution: None | numpy.typing.NDArray) -> tuple[list[float], list[float]]:


    wassersteins: list[float] = []
    kls = []
        
    for i, (ax, subtitle) in enumerate(zip(axs, subtitles)):
        ax.spines['right'].set_position(('outward', 80))
        ax.tick_params('y')

        ax.set_xlabel(f"{component_names} %d" % (i))

        xmin = numpy.min(model_projected_data[:, i])
        xmax = numpy.max(model_projected_data[:, i])
        xmin = np.min([xmin, np.min(native_proj_datas_concat[:,i])])
        xmax = np.max([xmax, np.max(native_proj_datas_concat[:,i])])
        
        Xs = numpy.linspace(xmin, xmax, num=100)
        model_kernel = scipy.stats.gaussian_kde(model_projected_data[:, i].T)
        Ys_model = model_kernel(Xs)
        ax.plot(Xs, Ys_model, c="red", label='model')
        
        # stride = 10 
        # some proteins have 30k points, some have 1M+ points (why is that? look into it as some point). stride more if 1M+ points
        native_kde = scipy.stats.gaussian_kde(native_proj_datas_concat[:, i].T)
        w1_tica1d = wasserstein1d(native_kde, model_kernel, xmin, xmax)
        wassersteins.append(w1_tica1d)
        kl = kl_div_calc(native_kde, model_kernel, native_proj_datas_concat[:, i].T)
        kls.append(kl)
    
        Ys_native = native_kde(Xs)
        ax.plot(Xs, Ys_native, c="blue", label='ground truth')
        if i == 0:
            if stationary_distribution is not None:
                num_bins = stationary_distribution.shape[1]
                ax.plot(stationary_distribution[0, :], stationary_distribution[1, :] * num_bins / (xmax - xmin), label="MSM equilibrium density", color="orange")
        ymax = max(Ys_model.max(), Ys_native.max())
        ax.set_ylim((0.0, 1.2* ymax))
        ax.set_title(subtitle)
    return wassersteins, kls


# some model trajs might explode, take those out for now
def check_if_traj_has_nans(protein_name: str, index: int, traj: ModelTraj):
    assert isinstance(traj.trajectory.xyz, np.ndarray)
    is_nan_list = np.isnan(traj.trajectory.xyz)
    if is_nan_list.any():
        first_frame = is_nan_list.any(axis=(1,2)).argmax(axis=0)
        logging.warning(f"{protein_name} model traj {index} has nans at frame {first_frame}/{is_nan_list.shape[0]}, removing it")
        return False
    else:
        return True
            

def plot_trajectory_raw(tica_coords: numpy.typing.NDArray, out_file: Path):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    stride = 1
    axes.scatter(tica_coords[::stride, 0], tica_coords[::stride, 1], c=tica_coords[::stride, 2], s=1)
    fig.savefig(out_file)
    plt.close(fig)
    
    

if __name__ == "__main__":
    main()
