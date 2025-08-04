import mdtraj
import numpy as np
from dataclasses import dataclass
from .tica_plots import calc_atom_distance
import numpy
import numpy.typing
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from report_generator.cache_loading import load_cache_or_make_new

colors = [
    (0, "blue"),    # Lowest value (-1)
    (0.5, "white"), # Middle value (0)
    (1, "red")      # Highest value (1)
]
custom_cmap = LinearSegmentedColormap.from_list("CustomMap", colors)
THRESHOLD = 1.5

@dataclass
class ContactMap:
    matrix: numpy.typing.NDArray
    n_atoms: int


def get_contact_maps(
        native_trajs: list[mdtraj.Trajectory],
        protein_name: str,
        cache_path: Path,
        force_cache_regen: bool,
        temperature: int) -> tuple[Path, ContactMap]:
    cache_filename = cache_path.joinpath(f"{protein_name}_{temperature}K.contact_map")
    make_new = lambda: make_contact_map(native_trajs)
    contact_map = load_cache_or_make_new(
        Path(cache_filename),
        make_new,
        ContactMap,
        force_cache_regen
    )
    return cache_filename, contact_map

#mostly copied from cyruses code notebook
def make_contact_map(trajs: list[mdtraj.Trajectory]) -> ContactMap:
    distances = np.concatenate(np.array([calc_atom_distance(x) for x in trajs]))
    less_than_threshold = distances < THRESHOLD
    percentages = numpy.sum(less_than_threshold, axis=0) / less_than_threshold.shape[0]
    return ContactMap(percentages, trajs[0].n_atoms)


def make_visual_matrix(contact_map: ContactMap) -> numpy.typing.NDArray:

    visualmatrix = np.ones([len(contact_map.matrix),len(contact_map.matrix)])
    c = 0
    for i in range(contact_map.n_atoms):
        for j in range(i):
            if i == j:
                continue
            visualmatrix[i][j] = contact_map.matrix[c]
            visualmatrix[j][i] = contact_map.matrix[c]
            c += 1
    return visualmatrix

def make_contact_map_plot(axes, native_contact_map: ContactMap, model_contact_map: ContactMap):
    assert native_contact_map.n_atoms == model_contact_map.n_atoms
    native_matrix = make_visual_matrix(native_contact_map)
    model_matrix = make_visual_matrix(model_contact_map)
    delta = model_matrix - native_matrix

    im = axes.imshow(delta, cmap=custom_cmap)
    axes.set_title(f"Contact map difference: Model - GT")
    axes.set_ylim(0,native_contact_map.n_atoms)
    axes.set_xlim(0,native_contact_map.n_atoms)
    fig = axes.figure
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
    #axes.set_layout_engine("tight")
    #return fig

# def make_contact_map_plot(axes, native_contact_map: ContactMap, model_contact_map: ContactMap) -> Figure:
#     assert native_contact_map.n_atoms == model_contact_map.n_atoms
#     native_matrix = make_visual_matrix(native_contact_map)
#     model_matrix = make_visual_matrix(model_contact_map)
#     delta = model_matrix - native_matrix
    
#     # fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)

#     for i, (title, matrix) in enumerate(zip(
#             ["native", "model", "delta"],
#             [native_matrix, model_matrix, delta])):
#         im = axes[0, i].imshow(matrix, cmap='magma')
#         fig.colorbar(im, ax=axes[0, i])
#         axes[0, i].set_title(title)
#         axes[0, i].set_ylim(0,native_contact_map.n_atoms)
#         axes[0, i].set_xlim(0,native_contact_map.n_atoms)
#     return fig
