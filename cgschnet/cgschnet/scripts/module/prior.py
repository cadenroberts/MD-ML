import numpy as np
import os
from scipy.optimize import curve_fit
import mdtraj
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


# original code https://github.com/torchmd/torchmd-cg/blob/master/torchmd_cg/utils/prior_fit.py

def make_key(at, tagged=False):
    if tagged:
        tagged = all([i.endswith("*") for i in at])
    at = [i.replace("*", "") for i in at]

    if len(at) > 1 and at[0] > at[-1]:
        result = tuple(reversed(at))
    else:
        result = tuple(at)

    if tagged:
        result = tuple([i+"*" for i in result])

    return result

def key_to_str(key):
    return "(" + ", ".join(key) + ")"

def CG(r,eps,V0):
    sigma=1.0
    sigma_over_r = sigma/r
    V = 4*eps*(sigma_over_r**6) + V0
    return V

def harmonic(x,x0,k,V0): 
    return k*(x-x0)**2+V0

kB = 0.0019872041 # kcal/mol/K

class ParamBondedCalculator:
    def __init__(self, unified=False, directional=False, fitSpecificBonds=None):
        self.bond_dists = {}
        self.prior_bond = {}
        self.unified = unified
        self.directional = directional
        self.bond_range = [0, 8.0]
        self.num_bins = 8*100 # Generate bins ~0.1 Ang wide
        self.bin_edges = np.linspace(self.bond_range[0], self.bond_range[1], self.num_bins + 1, dtype=np.float32)
        self.fitSpecificBonds = fitSpecificBonds

    def renorm_bonds(self, counts, bins):
        R = .5*(bins[1:]+bins[:-1])  # bin centers
        vols = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)
        ncounts = counts / vols
        return np.vstack([R,ncounts])

    def add_molecule(self, mol, traj, cache_dir=None):
        """Add the bonds from "mol" for future fit calculations"""

        # get bonds types and indices
        # ex
        # ('CAP', 'CAQ'): [array([0, 1], dtype=uint32),
        #  array([ 99, 100], dtype=uint32)]
        bonds_types = defaultdict(list)
        if self.unified:
            bonds_types[("X","X")].extend(mol.bonds)
        else:
            for bond in mol.bonds:
                if self.directional:
                    bonds_types[tuple(mol.atomtype[bond])].append(bond)
                else:
                    bonds_types[make_key(mol.atomtype[bond], tagged=True)].append(bond)

        hists = {}
        for bond in bonds_types.keys():
            # Periodic is false here because bonded atoms will always be in the same box 
            # OPENMM HAS NO BONDS ACROSS BOXES, MIGHT BE DIFFERENT FOR OTHER METHODS
            traj_dists = mdtraj.compute_distances(traj, bonds_types[bond], periodic=False).flatten()
            hist, _ = np.histogram(traj_dists, bins=self.bin_edges)
            assert np.sum(hist) == len(traj_dists), "Out of range value"
            hists[key_to_str(bond)] = hist

        if cache_dir:
            np.savez(os.path.join(cache_dir, "bonds.npz"), **hists)

        self.merge_hists(hists)

    def load_molecule_cache(self, cache_dir):
        hists = np.load(os.path.join(cache_dir, "bonds.npz"))
        self.merge_hists(hists)

    def merge_hists(self, hists):
        bondList = self.fitSpecificBonds if self.fitSpecificBonds else list(hists.keys())
        for bond in bondList:
            if bond not in self.bond_dists:
                self.bond_dists[bond] = np.zeros(self.num_bins)
            self.bond_dists[bond] += hists[bond]

    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0):
        """Calculate bond parameters"""

        for name, dists in self.bond_dists.items():
            # normalize distance counts by spherical shell volume
            RR, ncounts = self.renorm_bonds(dists, self.bin_edges)
            # Drop zero counts
            RR_nz = RR[ncounts > min_cnt]
            ncounts_nz = ncounts[ncounts > min_cnt]
            dG_nz = -1*kB*Temp*np.log(ncounts_nz)

            seen_range = [np.min(RR_nz), np.max(RR_nz)]

            if fit_constraints:
                fit_bounds = [[seen_range[0],0,-np.inf], [seen_range[1],np.inf,np.inf]]
            else:
                fit_bounds = (-np.inf, np.inf)

            popt, _ = curve_fit(harmonic, RR_nz, dG_nz,
                                p0=[np.array(seen_range).mean(), 60, -1],
                                bounds=fit_bounds,
                                maxfev=100000)

            self.prior_bond[name] = {'req': popt[0].tolist(),
                                      'k0':  popt[1].tolist() }

            if plot_directory:
                # https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
                plt.figure() # Make a new plot
                plt.plot(RR_nz, dG_nz, 'o')
                plot_space = np.linspace(*np.array(seen_range), 100) #pyright: ignore[reportCallIssue]
                plt.plot(plot_space, harmonic(plot_space, *popt))
                plot_name = name
                plt.xlabel('distance (A)')
                plt.ylabel('dG (kcal/mol)')
                plt.title(plot_name)
                plt.savefig(os.path.join(plot_directory, f'{plot_name}-fit.png'))
                plt.close() # Don't leak the old one
        
        return self.prior_bond


class NullParamBondedCalculator:
    """This class generates a zero energy bond prior, which can be used to disable prior bonds
    without breaking TorchMD's non-bonded exclusion logic."""
    def __init__(self):
        pass

    def add_molecule(self, mol, traj, cache_dir=None):
        """Noop"""

    def load_molecule_cache(self, cache_dir):
        """Noop"""

    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0):
        """Return a zero force bond prior"""
        prior_bond = {
            "(X, X)": {"req":4.0, "k0": 0.0}
        }

        return prior_bond

class NullParamAngleCalculator:
    """This class generates a zero energy angle prior, which can be used to disable prior bonds
    without breaking TorchMD's non-bonded exclusion logic."""
    def __init__(self):
        pass

    def add_molecule(self, mol, traj, cache_dir=None):
        """Noop"""

    def load_molecule_cache(self, cache_dir):
        """Noop"""

    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0):
        """Return a zero force angle prior"""
        prior = {
            "(X, X, X)": {"theta0":120.0, "k0": 0.0}
        }

        return prior

class NullParamDihedralCalculator:
    """This class generates a zero energy angle prior, which can be used to disable prior bonds
    without breaking TorchMD's non-bonded exclusion logic."""
    def __init__(self):
        pass

    def add_molecule(self, mol, traj, cache_dir=None):
        """Noop"""

    def load_molecule_cache(self, cache_dir):
        """Noop"""

    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0):
        """Return a zero force dihedral prior"""
        prior = {
            "(X, X, X, X)": {"terms": [
                {
                    "phi_k": 0.0,
                    "phase": 0.0,
                    "per": 1,
                }
            ]}
        }

        return prior

#FIXME: this renorm is a duplicate of the ParamBondedCalculator version above
# Input:  counts and bin limits (n+1)
# Output: counts normalized by spheric shell volumes, and bin centers (n)
def renorm_nonbonded(counts, bins):
    R = .5*(bins[1:]+bins[:-1])  # bin centers
    vols = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)
    ncounts = counts / vols
    return np.vstack([R,ncounts])

class ParamNonbondedCalculator:
    def __init__(self, fit_range, exclusion_terms = None):
        self.atom_dists = {}
        self.prior_lj = {}
        self.fit_range = fit_range
        self.num_bins = 30
        self.bin_edges = np.linspace(self.fit_range[0], self.fit_range[1], self.num_bins + 1, dtype=np.float32)
        if exclusion_terms is not None:
            self.exclusion_terms = set(exclusion_terms)
        else:
            # FIXME: The dihedrals term is named incorrectly, it should be called 1-4
            self.exclusion_terms = {"bonds", "angles", "dihedrals"}
        assert self.exclusion_terms - {"bonds", "angles", "dihedrals", "1-4"} == set(), "Unknown exclusion terms"

    def add_molecule(self, mol, traj, cache_dir=None):
        """Add "mol" for future fit calculations"""

        atom_types = {}
        # Don't use tags for non-bonded interactions
        atom_type_keys = np.array([i.replace("*","") for i in mol.atomtype], dtype=object)
        for at in set(atom_type_keys):
            atom_types[at] = np.where(atom_type_keys == at)[0]

        hists = {}
        for at in atom_types.keys():
            # For each atom of this type
            for idx in atom_types[at]:
                # Make a list of everything that should be excluded from the JL calculation

                exclusions = [idx]

                # Other atoms bonded to this one
                if "bonds" in self.exclusion_terms:
                    for bond in mol.bonds:
                        # Just include both side of the bond
                        if idx in bond:
                            exclusions.extend(bond)

                # Or involved in an angle
                if "angles" in self.exclusion_terms:
                    for angle in mol.angles:
                        if idx in angle:
                            exclusions.extend(angle)

                # Or dihedral
                # FIXME: "dihedrals" is not a valid exclusions term
                if ("dihedrals" in self.exclusion_terms) or ("1-4" in self.exclusion_terms):
                    for dihedral in mol.dihedrals:
                        if idx in dihedral:
                            exclusions.extend(dihedral)

                computeTogether = []
                for idx2 in range(mol.numAtoms):
                    if idx2 not in exclusions:
                        computeTogether.append([idx, idx2])

                if len(computeTogether) == 0:
                    raise RuntimeError(f"No non-bonded interactions found for atom type: {at}")

                traj_dists = mdtraj.compute_distances(traj, computeTogether).flatten()
                # Slightly faster to trim things first
                traj_dists = traj_dists[traj_dists<self.fit_range[1]]
                hist, _ = np.histogram(traj_dists, bins=self.bin_edges)
                # Unlike the other fits we expect to the range to discard values
                # assert np.sum(hist) == len(traj_dists), "Out of range value"

                # Note that for this one we do accumulate values rather than calculating a single block per type
                if at not in hists:
                    hists[at] = np.zeros(self.num_bins)
                hists[at] += hist

        if cache_dir:
            np.savez(os.path.join(cache_dir, "lj.npz"), **hists)

        self.merge_hists(hists)

    def load_molecule_cache(self, cache_dir):
        hists = np.load(os.path.join(cache_dir, "lj.npz"))
        self.merge_hists(hists)

    def merge_hists(self, hists):
        for at in hists.keys():
            if at not in self.atom_dists:
                self.atom_dists[at] = np.zeros(self.num_bins)
            self.atom_dists[at] += hists[at]

    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0):
        for at, dists in self.atom_dists.items():
            RR, ncounts = renorm_nonbonded(dists, self.bin_edges)

            RR_nz = RR[ncounts > min_cnt]
            ncounts_nz = ncounts[ncounts > min_cnt]
            dG_nz = -kB * Temp * np.log(ncounts_nz)

            popt, _ = curve_fit(CG, RR_nz, dG_nz, p0=[2229.0, 1.], maxfev=100000)

            self.prior_lj[at] = {'epsilon': popt[0].tolist(),
                                'sigma': 1.0}

            if plot_directory:
                plt.figure() # Make a new plot
                # FIXME: Use a np.linspace for the curve plot
                plt.plot(RR_nz, dG_nz, 'o')
                plt.plot(RR_nz, CG(RR_nz, *popt))
                plot_name = f'{at}'
                plt.title(plot_name)
                plt.savefig(os.path.join(plot_directory, f'non-bonded-{plot_name}-fit.png'))
                plt.close() # Don't leak the old one

        return self.prior_lj

def renorm_angles(counts, bins):
    # The normalization term if we were integrating would be 1/sin(theta) to
    # account for the decreasing width of the volume element as the angle moves
    # towards the pole. The discrete version of this is the integral over the bin,
    # giving us cos(thetaEnd)-cos(thetaStart).
    # https://doi.org/10.1002/(SICI)1521-4044(199802)49:2/3%3C61::AID-APOL61%3E3.0.CO;2-V

    R = .5*(bins[1:]+bins[:-1]) # bin centers

    cos_vals = np.cos(bins)
    vols = (cos_vals[:-1]-cos_vals[1:])
    ncounts = counts / vols

    return np.vstack([R,ncounts])

class ParamAngleCalculator:
    def __init__(self, center=False, fitSpecificAngles=None):
        self.thetas = {}
        self.prior_angle = {}
        self.center = center
        self.angle_range = [0, np.pi]
        self.num_bins = 40
        # dtype=np.float32 is not really necessary, but it allows the results to exactly match what you'd get
        # by letting np.histogram define the binds automatically.
        self.bin_edges = np.linspace(self.angle_range[0], self.angle_range[1], self.num_bins + 1, dtype=np.float32)
        self.fitSpecificAngles=fitSpecificAngles

    def add_molecule(self, mol, traj, cache_dir=None):
        """Add "mol" for future fit calculations"""
        angles_types = defaultdict(list)

        if self.center:
            for angle in mol.angles:
                key = make_key(("X", mol.atomtype[angle[1]], "X"))
                angles_types[key].append(angle)
        else:
            for angle in mol.angles:
                angles_types[make_key(mol.atomtype[angle])].append(angle)

        hists = {}
        for angle in angles_types.keys():
            # Calculate the value (in radians) for each angle in the prior
            traj_angles = mdtraj.compute_angles(traj, angles_types[angle], periodic=False).flatten()
            hist, _ = np.histogram(traj_angles, bins=self.bin_edges)
            assert np.sum(hist) == len(traj_angles), "Out of range value"
            hists[key_to_str(angle)] = hist

        if cache_dir:
            np.savez(os.path.join(cache_dir, "angles.npz"), **hists)

        self.merge_hists(hists)

    def load_molecule_cache(self, cache_dir):
        hists = np.load(os.path.join(cache_dir, "angles.npz"))
        self.merge_hists(hists)

    def merge_hists(self, hists):
        angleList = self.fitSpecificAngles if self.fitSpecificAngles else list(hists.keys())
        for angle in angleList:
            # Calculate the value (in radians) for each angle in the prior
            if angle not in self.thetas:
                self.thetas[angle] = np.zeros(self.num_bins)
            self.thetas[angle] += hists[angle]


    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0):
        for name, thetas in self.thetas.items():
            RR, ncounts = renorm_angles(thetas, self.bin_edges)

            # Drop zero counts
            RR_nz = RR[ncounts > min_cnt]
            ncounts_nz = ncounts[ncounts > min_cnt]
            dG_nz = -kB*Temp*np.log(ncounts_nz)

            if fit_constraints:
                fit_bounds = [[0,0,-np.inf], [np.pi,np.inf,np.inf]]
            else:
                fit_bounds = (-np.inf, np.inf)

            # Angle values are in degrees
            popt, _ = curve_fit(harmonic, RR_nz, dG_nz,
                                p0=[np.pi/2, 60, -1],
                                bounds=fit_bounds,
                                maxfev=100000)

            # popt now has the function parameters
            # As of TorchMD 1.0.2 theta0 is in degrees but k0 is in energy/radians^2
            self.prior_angle[name] = {'theta0': popt[0].tolist() * 180.0/np.pi,
                                      'k0':  popt[1].tolist() }

            if plot_directory is not None:
                plt.figure() # Make a new plot
                plt.plot(RR_nz, dG_nz, 'o')
                plot_space = np.linspace(*self.angle_range, 100) #pyright: ignore[reportCallIssue]
                plt.plot(plot_space, harmonic(plot_space, *popt))
                plot_name = name
                plt.title(plot_name)
                plt.savefig(os.path.join(plot_directory, f'angle-{plot_name}-fit.png'))
                plt.close() # Don't leak the old one

        return self.prior_angle
    

class ParamDihedralCalculator:
    def __init__(self, terms=2, unified=False, directional=False, scale=1.0):
        self.thetas = {}
        self.prior_dihedral = {}
        self.n_terms = terms
        self.unified = unified
        self.directional = directional
        self.scale = scale
        self.dihedral_range = [-np.pi, np.pi]
        self.num_bins = 80
        self.bin_edges = np.linspace(self.dihedral_range[0], self.dihedral_range[1], self.num_bins + 1, dtype=np.float32)

    def add_molecule(self, mol, traj, cache_dir=None):
        """Add "mol" for future fit calculations"""

        hists = {}
        if self.unified:
            key = ("X","X","X","X")
            traj_dihedrals = mdtraj.compute_dihedrals(traj, mol.dihedrals).flatten()
            hist, _ = np.histogram(traj_dihedrals, bins=self.bin_edges)
            assert np.sum(hist) == len(traj_dihedrals), "Out of range value"
            hists[key_to_str(key)] = hist
        else:
            dihedrals_types = defaultdict(list)
            for dihedral in mol.dihedrals:
                if self.directional:
                    dihedrals_types[tuple(mol.atomtype[dihedral])].append(dihedral)
                else:
                    dihedrals_types[make_key(mol.atomtype[dihedral])].append(dihedral)

            for dihedral in dihedrals_types.keys():
                # mdtraj.compute_dihedrals returns data as a [n_dihedrals, n_frames] array
                traj_dihedrals = mdtraj.compute_dihedrals(traj, dihedrals_types[dihedral]).flatten()
                hist, _ = np.histogram(traj_dihedrals, bins=self.bin_edges)
                assert np.sum(hist) == len(traj_dihedrals), "Out of range value"
                hists[key_to_str(dihedral)] = hist
        self.merge_hists(hists)

        if cache_dir:
            np.savez(os.path.join(cache_dir, "dihedrals.npz"), **hists)

    def load_molecule_cache(self, cache_dir):
        hists = np.load(os.path.join(cache_dir, "dihedrals.npz"))
        self.merge_hists(hists)

    def merge_hists(self, hists):
        for dihedral in hists.keys():
            if dihedral not in self.thetas:
                self.thetas[dihedral] = np.zeros(self.num_bins)
            self.thetas[dihedral] += hists[dihedral]

    def dihedral_fit_fun(self, theta, offset, *args):
        # Implements the TorchMD torsion function
        # https://doi.org/10.1021/acs.jctc.0c01343?rel=cite-as&ref=PDF&jav=VoR
        # args = [phi_k0, phase0, phi_k1, phase1, ...]
        assert len(args) == self.n_terms*2
        result = offset
        for i in range(0,self.n_terms):
            phi_k = args[i*2]
            phase = args[i*2+1]
            per = i+1
            # result += phi_k*(1+np.cos( (per*theta - phase)*np.pi/180.0) )
            result += phi_k*(1+np.cos(per*theta - phase) )
        return result

    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0):
        for name, thetas in self.thetas.items():
            # Dihedrals don't require normalization (all the binds are the same size),
            # but we still need to convert to degrees
            # RR = .5*(self.bin_edges[1:]+self.bin_edges[:-1])*180/np.pi  # bin centers
            
            # Raz: I changed this to radians, and converted to degrees at the end of the function when saving the values, to harmonize with the way the angle terms work. It shouldn't affect any downstream functionality, but made it easier to test that the NN fit was working correctly. As the NN prior is trained in with radian angles.
            RR = .5*(self.bin_edges[1:]+self.bin_edges[:-1])  # bin centers
            ncounts = thetas

            # Drop zero counts
            RR_nz = RR[ncounts > min_cnt]
            ncounts_nz = ncounts[ncounts > min_cnt]
            dG_nz = -kB*Temp*np.log(ncounts_nz)

            # Fit may fail, better to try-catch. p0 usually not necessary if function is reasonable.
            p0: list[float] = [0] # The first parameter is an arbitrary offset from zero
            for i in range(self.n_terms):
                p0.append(0.1)
                p0.append(i/self.n_terms)

            popt, _ = curve_fit(self.dihedral_fit_fun, RR_nz, dG_nz, p0=p0, maxfev=100000, xtol=1e-10, ftol=1e-10)
            
            # popt now has the function parameters
            terms = []
            for i in range(self.n_terms):
                # +1 and +2 because we skip the offset term in popt
                popt[i*2+1] *= self.scale
                terms.append({
                    "phi_k": popt[i*2+1].tolist(),
                    "phase": popt[i*2+2].tolist() * 180.0/np.pi,
                    "per": i+1,
                })

            self.prior_dihedral[name] = {'terms': terms, 'offset': popt[0].tolist()}

            if plot_directory is not None:
                plt.figure() # Make a new plot
                plt.plot(RR_nz, dG_nz, 'o')
                plot_space = np.linspace(*(np.array(self.dihedral_range)), 100) #pyright: ignore[reportCallIssue]
                plt.plot(plot_space, self.dihedral_fit_fun(plot_space, *popt))
                plot_name = name
                plt.title(plot_name)
                plt.savefig(os.path.join(plot_directory, f'dihedral-{plot_name}-fit.png'))
                plt.close() # Don't leak the old one

        return self.prior_dihedral

