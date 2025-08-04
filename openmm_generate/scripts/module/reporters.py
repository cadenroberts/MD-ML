import numpy as np
import h5py

from openmm.app import *
from openmm import *
from openmm.unit import *

import json
import mdtraj
from mdtraj.utils import unitcell

import time
import threading
import queue

# The following code is taken from mdtraj/mdtraj/formats/hdf5.py and is LGPL licenced:
##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2013 Stanford University and the Authors
#
# Authors: Robert McGibbon
# Contributors:
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

def topology_to_H5MD_json(openmm_topology, atom_indices=None):
    topology_object = mdtraj.Topology.from_openmm(openmm_topology)
    
    if atom_indices:
        topology_object = topology_object.subset(atom_indices)

    try:
        topology_dict = {
            'chains': [],
            'bonds': []
        }

        for chain in topology_object.chains:
            chain_dict = {
                'residues': [],
                'index': int(chain.index)
            }
            for residue in chain.residues:
                residue_dict = {
                    'index': int(residue.index),
                    'name': str(residue.name),
                    'atoms': [],
                    "resSeq": int(residue.resSeq),
                    "segmentID": str(residue.segment_id)
                }

                for atom in residue.atoms:

                    try:
                        element_symbol_string = str(atom.element.symbol)
                    except AttributeError:
                        element_symbol_string = ""

                    residue_dict['atoms'].append({
                        'index': int(atom.index),
                        'name': str(atom.name),
                        'element': element_symbol_string
                    })
                chain_dict['residues'].append(residue_dict)
            topology_dict['chains'].append(chain_dict)

        for atom1, atom2 in topology_object.bonds:
            topology_dict['bonds'].append([
                int(atom1.index),
                int(atom2.index)
            ])

    except AttributeError as e:
        raise AttributeError('topology_object fails to implement the'
            'chains() -> residue() -> atoms() and bond() protocol. '
            'Specifically, we encountered the following %s' % e)

    data = json.dumps(topology_dict).encode('ascii')
    # if not isinstance(data, bytes):
    #     data = data.encode('ascii')

    return data

# The following code was taken from mdtraj/mdtraj/reporters/basereporter.py and is LGPL licenced:

def generate_cell_info(state, units):
    vectors = state.getPeriodicBoxVectors(asNumpy=True)
    # vectors = vectors.value_in_unit(getattr(units, self._traj_file.distance_unit))
    vectors = vectors.value_in_unit(units)
    a, b, c, alpha, beta, gamma = unitcell.box_vectors_to_lengths_and_angles(*vectors)
    result = {
        'cell_lengths' : np.array([a, b, c]),
        'cell_angles'  : np.array([alpha, beta, gamma])
    }
    return result

# End of mdtraj code

class ExtendedH5MDReporter:
    """
    A threaded h5py reporter based on H5MDReporter that saves forces data.
    """

    def __init__(self, filename, report_interval, total_steps, atom_subset=None, use_gzip=False, append_file=False):
        """
        Parameters:
        - filename (str): The path the save the H5MD file to.
        - report_interval (int): Write an entry every 'report_interval' steps.
        - total_steps (int): The total number of steps to be simulated.
        - atom_subset (list): The subset of atoms to record, or None for all atoms.

        Returns:
        None
        """
        self.report_interval = report_interval
        self.total_steps = total_steps
        self.atom_subset = atom_subset
        self.use_gzip = use_gzip

        if append_file and os.path.exists(filename):
            self.h5 = h5py.File(filename, 'a') # Create or continue existing file
            self.h5_initialized = True
            if self.h5["time"].size == self.h5["time"].maxshape[0]:
                raise RuntimeError(f"Can't continue '{filename}', the file is already full")
        else:
            self.h5 = h5py.File(filename, 'w') # Create or truncate file
            self.h5_initialized = False

        self._worker_thread_queue = queue.Queue(2) # Create a 2 deep work queue
        self.worker_thread = threading.Thread(target=self._worker_thread_run)
        self.worker_thread.start()

    def add_attr_string(self, obj, name, value):
        """
        Adds a PyTables compatible attribute string to obj.

        Parameters:
        - obj (h5py.Group or h5py.Dataset): The HDF5 object to which the attribute will be added.
        - name (str): The name of the attribute.
        - value (str): The value of the attribute.

        Returns:
        None
        """
        # Note that we have to ensure all strings are fixed length for mdtraj to open our files
        obj.attrs.create(name, value, dtype=h5py.string_dtype('ascii', len(value)))

    def write_str_dataset(self, name, str_data):
        """
        Adds a PyTables compatible string dataset to the H5MD file.

        Parameters:
        - name (str): The name of the dataset.
        - str_data (str): The string data.

        Returns:
        None
        """
        # Writes a string dataset in a format that pytables / mdtraj will understand
        self.h5.create_dataset(name, data=[str_data], dtype=h5py.string_dtype('ascii', len(str_data)), compression="gzip")

    def new_dataset(self, name, data, units):
        """
        Adds a new dataset in the H5MD file.

        Parameters:
        - name (str): The name of the dataset.
        - data (np.ndarray): The data to be stored in the dataset.
        - units (str): The units string for the data.

        Returns:
        None
        """
        kwargs = {}
        if self.use_gzip:
            kwargs["shuffle"] = True
            kwargs["compression"] = "gzip"

        chunk_shape = tuple([10] + list(data.shape)[1:])
        max_shape = tuple([self.total_steps] + list(data.shape)[1:])

        self.h5.create_dataset(name, data=data, chunks=chunk_shape, maxshape=max_shape, **kwargs)
        self.add_attr_string(self.h5[name], "units", units)

    def append_data(self, name, data):
        """
        Appends an array of data to an existing dataset in the H5MD file. To append one entry
        use pass data[np.newaxis,:].

        Parameters:
        - name (str): The name of the dataset.
        - data (np.ndarray): The data to be appended.

        Returns:
        None
        """
        self.h5[name].resize((self.h5[name].shape[0] + data.shape[0]), axis = 0)
        self.h5[name][-data.shape[0]:] = data

    def init_metadata(self):
        """
        Initialize the H5MD metadata with default values.

        Returns:
        None
        """
        self.add_attr_string(self.h5, "application", b'ExtendedH5MDReporter')
        self.add_attr_string(self.h5, "conventionVersion", b'1.1')
        self.add_attr_string(self.h5, "conventions", b'Pande')
        self.add_attr_string(self.h5, "program", b'ExtendedH5MDReporter')
        self.add_attr_string(self.h5, "programVersion", b'0.0.0')
        self.add_attr_string(self.h5, "title", b'none')

    def describeNextReport(self, simulation):
        # Returns a tuple describing when we want OpenMM to call us next and what data to provide
        return (self.report_interval,
                True,  # positions
                False, # velocities
                True,  # forces
                True,  # energies
                None   # Wrap to periodic box (None=True if the simulation is in a box, False if not)
                )

    def report(self, simulation, state):
        # Called by OpenMM with the data requested in self.describeNextReport()
        positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        forces = state.getForces(asNumpy=True).value_in_unit(kilojoules/mole/nanometer)
        cell_info = generate_cell_info(state, nanometer)
        time = state.getTime().value_in_unit(picoseconds)
        potentialEnergy = state.getPotentialEnergy().value_in_unit(kilojoules/mole)
        kineticEnergy = state.getKineticEnergy().value_in_unit(kilojoules/mole)

        if self.atom_subset is not None:
            positions = positions[self.atom_subset]
            forces = forces[self.atom_subset]

        positions = positions[np.newaxis,:].astype(np.float32)
        forces = forces[np.newaxis,:].astype(np.float32)

        cell_lengths = cell_info["cell_lengths"][np.newaxis,:].astype(np.float32)
        cell_angles  = cell_info["cell_angles"][np.newaxis,:].astype(np.float32)

        time = np.array([time], dtype=np.float32)
        potentialEnergy = np.array([potentialEnergy], dtype=np.float32)
        kineticEnergy = np.array([kineticEnergy], dtype=np.float32)

        if not self.h5_initialized:
            self.h5_initialized = True

            topology_json = topology_to_H5MD_json(simulation.topology, atom_indices=self.atom_subset.tolist())
            self.write_str_dataset("topology", topology_json)
            self.init_metadata()

            self.new_dataset("coordinates", positions, "nanometers")
            self.new_dataset("forces", forces, "kilojoules/mole/nanometer")
            self.new_dataset("cell_lengths", cell_lengths, "nanometer")
            self.new_dataset("cell_angles", cell_angles, "degrees")
            self.new_dataset("time", time, "picoseconds")
            self.new_dataset("potentialEnergy", potentialEnergy, "kilojoules/mole")
            self.new_dataset("kineticEnergy", kineticEnergy, "kilojoules/mole")
        else:
            data = {
                "coordinates" : positions,
                "forces" : forces,
                "cell_lengths" : cell_lengths,
                "cell_angles" : cell_angles,
                "time" : time,
                "potentialEnergy" : potentialEnergy,
                "kineticEnergy" : kineticEnergy,
            }

            self._worker_thread_queue.put(data)

    def _process_data(self, data):
        # Called by the worker thread to append data to the h5 file
        for k,v in data.items():
            self.append_data(k, v)

    def _worker_thread_run(self):
        # Main function for the worker thread
        while True:
            data = self._worker_thread_queue.get()
            if not data:
                break
            self._process_data(data)

    def close(self):
        """
        Clean up and close the H5MD file.

        Returns:
        None
        """
        if self.worker_thread:
            t0 = time.time()
            print(f"{self.__class__.__name__}: Waiting for worker thread to finish...")
            self._worker_thread_queue.put(None) # Signal the thread to terminate
            self.worker_thread.join()
            print(f"{self.__class__.__name__}: Worker thread done in {round(time.time()-t0, 4)} seconds.")
            self.worker_thread = None
        self.h5.close()

    def __del__(self):
        self.close()