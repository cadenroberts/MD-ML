import torch
import numpy as np
import os
from torch.utils.data import Dataset
from moleculekit.molecule import Molecule
from typing import Optional, BinaryIO

class NumpyReader:
    filename: str
    keep_open: bool
    file: Optional[BinaryIO]
    start_offset: int
    chunk_cnt: int
    chunk_bytes: int
    shape: tuple
    dtype: np.dtype

    def __init__(self, filename, keep_open=False):
        self.filename = filename
        self.keep_open = keep_open
        self.file = None

        # Open the file and read the header
        with open(self.filename, "rb") as f:
            magic = np.lib.format.read_magic(f)
            assert magic == (1, 0)
            self.shape, fortran_order, self.dtype = np.lib.format.read_array_header_1_0(f)[:3]
            assert not fortran_order, f"Can't load fortran order arrays: {self.filename}"
            self.start_offset = f.tell()
            self.chunk_cnt = int(np.prod(self.shape[1:]))
            self.chunk_bytes = self.chunk_cnt * self.dtype.itemsize

        # Keep the file open if specified
        if self.keep_open:
            self.file = open(self.filename, "rb")

    def __getitem__(self, index: int):
        if index < 0 or index >= self.shape[0]:
            raise IndexError("Index out of bounds")

        # Open the file if not already open
        if self.file is None:
            with open(self.filename, "rb") as f:
                f.seek(self.start_offset + self.chunk_bytes * index)
                return np.fromfile(f, self.dtype, self.chunk_cnt).reshape(self.shape[1:])
        else:
            self.file.seek(self.start_offset + self.chunk_bytes * index)
            return np.fromfile(self.file, self.dtype, self.chunk_cnt).reshape(self.shape[1:])

    def __len__(self):
        return int(self.shape[0])

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

class ProteinBatchCollate:
    def __init__(self, atoms_per_call):
        self.atoms_per_call = atoms_per_call

    def __call__(self, batch):
        def make_sub_batch(batch, batch_slice, lengths):
            sub_batch = {}
            for k in batch[0].keys():
                sub_batch[k] = torch.stack([torch.cat(j[k][batch_slice]) for j in batch])
            sub_batch["lengths"] = lengths
            return sub_batch

        atoms_per_call = self.atoms_per_call
        if atoms_per_call is None:
            atoms_per_call = float("inf")

        # Group the lengths into chunks <= atoms_per_call
        result = []
        cnt = 0
        batch_size = len(batch)
        start = 0
        group_lengths = []
        for i in range(len(batch[0]["pos"])):
            n_atoms = len(batch[0]["pos"][i])
            next_cnt = n_atoms*batch_size
            assert next_cnt <= atoms_per_call, f"Molecule {i} is too large ({n_atoms}x{batch_size})>{atoms_per_call}"
            if cnt + next_cnt > atoms_per_call:
                result.append(make_sub_batch(batch, slice(start, start+len(group_lengths)), group_lengths))
                # Reset for the next one
                start = i
                cnt = 0
                group_lengths = []
            group_lengths.append(n_atoms)
            cnt += next_cnt
        # Collect the remaining elements
        result.append(make_sub_batch(batch, slice(start, start+len(group_lengths)), group_lengths))
        return result

def make_batch_nums(repeats, lengths):
    result = []
    for j in range(repeats):
        for i, n in enumerate(lengths):
            k = i + j*len(lengths)
            result.append(torch.as_tensor(k, dtype=torch.long).repeat(n))
    return torch.cat(result)

def build_sequence_for_mol(mol):
    # This assumes that segid is an integer, which should be the case for all our psf files
    sequence = np.array([int(i)*20 for i in mol.segid]) + mol.resid
    assert(len(sequence) == mol.numAtoms)
    return sequence

def build_classical_terms_for_mol(mol, topology=None):
    if topology:
        bonds = np.empty([0, 2], dtype=np.int64)
        angles = np.empty([0, 3], dtype=np.int64)
        dihedrals = np.empty([0, 4], dtype=np.int64)
        # convert from uint32 to int64 for compatibility with torch
        if len(topology.bonds):
            bonds = np.array(topology.bonds, dtype=np.int64)
        if len(topology.angles):
            angles = np.array(topology.angles, dtype=np.int64)
        if len(topology.dihedrals):
            dihedrals = np.array(topology.dihedrals, dtype=np.int64)
    else:
        segid = np.array([int(i) for i in mol.segid])
        resid = np.array([int(i) for i in mol.resid])
        # Assert things are a simple sequence if not using a topology file
        assert np.all(segid[1:] >= segid[:-1])
        assert np.all(segid == segid[0])
        assert np.all(resid[1:] == resid[:-1] + 1)

        idx = np.arange(len(segid))
        bonds = np.array([idx[:-1], idx[1:]]).T
        angles = np.array([idx[:-2], idx[1:-1], idx[2:]]).T
        dihedrals = np.array([idx[:-3], idx[1:-2], idx[2:-1],idx[3:]]).T

    return bonds, angles, dihedrals

class ProteinDataset(Dataset):
    """
    This class provides a Dataset that can pull from multiple trajectories at once and
    arrange the data into batches appropriate for passing to TorchMD.
    """

    def __init__(self, directory, pdb_ids, forces_file='deltaforces.npy', energy_file=None, embeddings_file="embeddings.npy", use_npfile=False):
        """
        Initialize the dataset.
        
        :param directory: The directory to find the prepared data files in.
        :param pdb_ids: A list of protein subdirectories to load from within "directory".
        """
        self.directory = directory
        self.pdb_ids = pdb_ids
        self.use_npfile = use_npfile
        self.coordinates = []
        self.embeddings = []
        self.deltaforces = []
        self.energies = []
        self.boxes = []
        self.sequences = None

        self.frame_terms = {}
        self.fixed_terms = {}

        for pdbid in pdb_ids:
            self.coordinates.append(self._load_dataset(f'{directory}/{pdbid}/raw/coordinates.npy'))
            self.embeddings.append(np.load(f'{directory}/{pdbid}/raw/{embeddings_file}'))
            self.deltaforces.append(self._load_dataset(f'{directory}/{pdbid}/raw/{forces_file}'))
            if energy_file is not None: 
                self.energies.append(np.expand_dims(np.load(f'{directory}/{pdbid}/raw/{energy_file}', mmap_mode="r"), axis=(1, 2)))

            if os.path.exists(f'{directory}/{pdbid}/raw/box.npy'):
                self.boxes.append(self._load_dataset(f'{directory}/{pdbid}/raw/box.npy'))

        # print("bb", np.expand_dims(self.boxes[0][0], 1).shape)

        if not self.boxes:
            self.boxes = None
        else:
            assert len(self.coordinates) == len(self.boxes), \
            "If any boxes are specified, all trajectories must have boxes."

        errCoord = []
        errDeltaForces = []
        errEmbeddings = [] 
        errBoxes = []
        for i in range(len(self.coordinates)):
            if len(self.coordinates[0]) != len(self.coordinates[i]):
                errCoord += [self.pdb_ids[i]]

            if len(self.coordinates[i]) != len(self.deltaforces[i]):
               errDeltaForces += [self.pdb_ids[i]]

            if len(self.coordinates[i][0]) != len(self.embeddings[i]):
                errEmbeddings += [self.pdb_ids[i]]

            if self.boxes:
                if len(self.coordinates[i]) != len(self.boxes[i]):
                    errBoxes += [self.pdb_ids[i]]
        
        if len(errCoord) > 0:
            print("Several trajectories do not have the same number of frames:", errCoord)

        if len(errDeltaForces) > 0:
            print("Several trajectories do not have the same number of frames:", errDeltaForces)
        
        if len(errEmbeddings) > 0:
            print("Several trajectories do not have the same number of frames:", errEmbeddings)

        if len(errBoxes) > 0:
            print("Several trajectories do not have the same number of frames:", errBoxes)
        
        with open(f'{directory}/result/train_errors.txt', 'w') as f:
            f.write(f"Several trajectories do not have the same number of frames: {errCoord}\n")
            f.write(f"Several trajectories do not have the same number of frames: {errDeltaForces}\n")
            f.write(f"Several trajectories do not have the same number of frames: {errEmbeddings}\n")
            f.write(f"Several trajectories do not have the same number of frames: {errBoxes}\n")

        #FIXME: This isn't compatible with chunking the datasets because there can be multiple ProteinDataset
        #       objects per directory.
        # subset_ok = set(pdb_ids) - set(errCoord) - set(errDeltaForces) - set(errEmbeddings) - set(errBoxes)
        # with open(f'{directory}/result/train_subset_ok.txt', 'w') as f:
        #     print(f'Saved the subset of pdbs that are consistent to {directory}/result/train_subset_ok.txt. You can rerun it with ----subsetpdbs=train_subset_ok.txt')
        #     # write a single pdb for each line
        #     for item in subset_ok:
        #         f.write("%s\n" % item)

        if len(errCoord) > 0 or len(errDeltaForces) > 0 or len(errEmbeddings) > 0 or len(errBoxes) > 0:
            raise ValueError("Inconsistent data in the dataset. Please fix your dataset or take out the unsuitable protein trajectories before continuing. ")

        # Make the static tensors
        self.embeddings = [torch.as_tensor(i) for i in self.embeddings]

    def _load_dataset(self, path):
        if self.use_npfile:
            return NumpyReader(path)
        else:
            return np.load(path, mmap_mode="r")

    def _to_tensor(self, array, dtype):
        if self.use_npfile:
            return torch.as_tensor(array, dtype=dtype)
        else:
            return torch.as_tensor(np.copy(array), dtype=dtype)

    def build_sequences(self):
        self.sequences = []
        for pdbid in self.pdb_ids:
            mol = Molecule(f'{self.directory}/{pdbid}/processed/{pdbid}_processed.psf')
            self.sequences.append(build_sequence_for_mol(mol))
        self.sequences = [torch.as_tensor(i, dtype=torch.long) for i in self.sequences]

    def load_frame_terms(self, names):
        """Load additional per-frame terms into the dataset. Each named term is loaded
        from 'name.npy' the 'raw' directory and will be returned in the batch under the same name."""

        for term_name in names:
            term_list = []
            for pdbid in self.pdb_ids:
                term_list.append(self._load_dataset(f"{self.directory}/{pdbid}/raw/{term_name}.npy"))
            self.frame_terms[term_name] = term_list

    def build_classical_terms(self):
        bonds_list = []
        angles_list = []
        dihedrals_list = []
        for pdbid in self.pdb_ids:
            mol = Molecule(f'{self.directory}/{pdbid}/processed/{pdbid}_processed.psf')
            topology = None
            if os.path.exists(f'{self.directory}/{pdbid}/processed/topology.psf'):
                topology = Molecule(f'{self.directory}/{pdbid}/processed/topology.psf')
            bonds, angles, dihedrals = build_classical_terms_for_mol(mol, topology)
            bonds_list.append(bonds)
            angles_list.append(angles)
            dihedrals_list.append(dihedrals)

        self.fixed_terms["bonds"] = [torch.as_tensor(i, dtype=torch.long) for i in bonds_list]
        self.fixed_terms["angles"] = [torch.as_tensor(i, dtype=torch.long) for i in angles_list]
        self.fixed_terms["dihedrals"] = [torch.as_tensor(i, dtype=torch.long) for i in dihedrals_list]

        self.fixed_terms["len_bonds"] = [torch.as_tensor([len(i)], dtype=torch.long) for i in bonds_list]
        self.fixed_terms["len_angles"] = [torch.as_tensor([len(i)], dtype=torch.long) for i in angles_list]
        self.fixed_terms["len_dihedrals"] = [torch.as_tensor([len(i)], dtype=torch.long) for i in dihedrals_list]

    def has_box(self):
        """
        Returns true if the dataset includes a periodic box
        """
        return self.boxes is not None

    def num_proteins(self):
        """
        Get the number of proteins in the dataset.
        """
        return len(self.coordinates)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.coordinates[0])

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index. If the
        dataset contains multiple proteins each item will contain one frame
        from every protein in the dataset.
        
        :param idx: Index of the data point to retrieve
        :return: A dict containing the data for frame "idx" of each protein
        """
        result = {}
        result["pos"] = [self._to_tensor(i[idx], dtype=torch.float32) for i in self.coordinates]
        result["z"] = self.embeddings
        result["force"] = [self._to_tensor(i[idx], dtype=torch.float32) for i in self.deltaforces]
        if len(self.energies) > 0:
            result["energy"] = [self._to_tensor(i[idx], dtype=torch.float32) for i in self.energies]

        if self.boxes:
                # TorchMD-Net expects the other values in a batch to be concatinated into a single list
                # (e.g. [Nmol*Natom,3] for coordinates), but boxes need to be distinct array entries
                # (e.g. [Nmol,3,3]). We add an extra dimention here so the same batch data handling functions
                # can be used for both types of objects.
            result["box"] = [self._to_tensor(np.expand_dims(i[idx], 0), dtype=torch.float32) for i in self.boxes]
        if self.sequences:
            result["s"] = self.sequences

        for k, v in self.fixed_terms.items():
            result[k] = v

        for k in self.frame_terms:
            result[k] = [self._to_tensor(i[idx], dtype=torch.float32) for i in self.frame_terms[k]]

        return result
