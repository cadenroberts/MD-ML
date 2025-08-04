#!/usr/bin/env python3
# pyright: reportIndexIssue=false, reportOptionalSubscript=false, reportCallIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportIncompatibleMethodOverride=false

import os
import numpy as np
import yaml
import json
from moleculekit.molecule import Molecule
from module import prior
# from module.make_deltaforces import DeltaForces
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmd.forces import Forces
from module.external_nn import ExternalNN, ParametersNN
import time
import argparse
from tqdm import tqdm
import glob
import pickle
import torch
from module.torchmd import tagged_forcefield
from preprocess import Preprocessor, Prior_CA
# from module.prior_flex import ParamDihedralCalculator, fitGPNN, NeuralNet, GPCustom
from module.prior_flex import ParamBondedFlexCalculator, kB, harmonic, curve_fit, ConstantKernel, RBF, WhiteKernel, GPCustom, NeuralNet, optim, fitGPNN, ParamAngleCalculator, renorm_angles, ParamDihedralFlexCalculator

# Set matplotlib to use a thread safe backend (for prior fit plots)
import matplotlib
matplotlib.use('Agg')


# Raz Dec 30 2024: turns out that when preprocessing the 6.5k dataset, the last 20 pdbs are taking forever to process due to being very big proteins. In addition, some were giving hdf5 errors since the batch_generate job didn't properly finish, and this means we don't have all the frames, and they will be removed later on before training. So here's the best workflow I found to solve it:
# 1) first run just step 1 on all proteins, and set FILTER_NOT_PROCESSED_STEP_ONE = False. if any pdbs are taking forever (generally the last 20), just kill the job
# 2) re-run the pre-processing for a second time with FILTER_NOT_PROCESSED_STEP_ONE = True 
FILTER_NOT_PROCESSED_STEP_ONE = True
# PDB_SUBSET=30
# USE_CACHED_PDB_LIST=True
# SKIP_STEP_ONE=False

def dihedral_fit_fun(theta, offset, *args):
        # Implements the TorchMD torsion function
        # https://doi.org/10.1021/acs.jctc.0c01343?rel=cite-as&ref=PDF&jav=VoR
        # args = [phi_k0, phase0, phi_k1, phase1, ...]
        # assert len(args) == 2*2
        # print('args', args)
        result = offset
        for i in range(0,2):
            phi_k = args[i*2]
            phase = args[i*2+1]
            per = i+1
            result += phi_k*(1+np.cos(per*theta - phase))
        return result

class Prior_CA_lj_angleXCX_dihedralX_specific(Prior_CA):
    """torchmd-cg CA prior with Bonded, Angle, DihedralX & RepulsionCG terms"""
    def __init__(self, fitSpecificBonds=None, fitSpecificAngles=None, forceterms=None):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angleXCX_dihedralX",
            "exclusions" : ['bonds', 'angles', 'dihedrals'],
            "forceterms" : forceterms if forceterms else ['bonds', 'angles', 'dihedrals'],
        })
        if 'bonds' in self.prior_params["forceterms"]:
            self.terms["bonds"] = prior.ParamBondedCalculator(fitSpecificBonds)
        
        # if 'lj' in self.prior_params["forceterms"]:
        #     self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        
        if 'angles' in self.prior_params["forceterms"]:
            self.terms["angles"] = prior.ParamAngleCalculator(center=True, fitSpecificAngles=fitSpecificAngles)
        
        if 'dihedrals' in self.prior_params["forceterms"]:
            self.terms["dihedrals"] = prior.ParamDihedralCalculator(unified=True)

class Prior_CA_lj_angleXCX_dihedralX_flex_specific(Prior_CA):
    """torchmd-cg CA prior with highly flexible Bonded, Angle, DihedralX & RepulsionCG terms that fit the data.

    """
    def __init__(self, fitSpecificBonds=None, fitSpecificAngles=None, forceterms_nn=None):
        super().__init__()
        self.prior_params.update({
            "prior_configuration_name": "CA_lj_angleXCX_dihedralX_flex",
            "exclusions" : ['bonds', 'angles', 'dihedrals'],
            "forceterms_classical" : [],
            "forceterms_nn" : forceterms_nn if forceterms_nn else ['bonds', 'angles', 'dihedrals'],
            "external" : True
        })
        self.prior_params['forceterms'] = self.prior_params['forceterms_classical'] + self.prior_params['forceterms_nn']
        if 'bonds' in self.prior_params["forceterms_nn"]:
            self.terms["bonds"] = ParamBondedFlexCalculatorDataFromPoly(fitSpecificBonds=fitSpecificBonds)
        
        # if 'lj' in self.prior_params["forceterms"]:
        #     self.terms["lj"] = prior.ParamNonbondedCalculator(fit_range=[3, 6])
        
        if 'angles' in self.prior_params["forceterms_nn"]:
            self.terms["angles"] = ParamAngleFlexCalculatorDataFromPoly(center=True, fitSpecificAngles=fitSpecificAngles)
        
        if 'dihedrals' in self.prior_params["forceterms_nn"]:
            self.terms["dihedrals"] = ParamDihedralCalculatorDataFromPoly(unified=True)

    # have to override this method since we're saving neural nets as priors
    def save_prior(self, output_path, pdbid):
        prefix = ""
        if pdbid:
            prefix = f"{pdbid}_"
        with open(os.path.join(output_path, f"{prefix}prior_params.json"),"w") as f:
            json.dump(self.prior_params, f)

        print('self.priors', self.priors.keys())
        # remove the dihedrals and bonds from the priors
        priorsTruncated = self.priors.copy()
        priorsTruncated.pop('dihedrals')
        priorsTruncated.pop('bonds')
        print('priorsTruncated', priorsTruncated.keys())

        with open(os.path.join(output_path, f"{prefix}priors.yaml"), "w") as f:
            yaml.dump(priorsTruncated, f)

        # save it with pickle instead
        with open(os.path.join(output_path, f"{prefix}priors.pkl"), "wb") as f:
            pickle.dump(self.priors, f)


def get_prior_params_path(prior_path):
    dir_path, file_name = os.path.split(prior_path)
    file_name = file_name.replace("priors.yaml", "prior_params.json")
    return os.path.join(dir_path, file_name)



class ParamBondedFlexCalculatorDataFromPoly(ParamBondedFlexCalculator):
    # if you only want to fit [(CAG, CAH), (CAG, CAL)] use fitSpeficBonds. If None, fit all
    def __init__(self, unified=False, fitSpecificBonds=None):
        super().__init__(unified)
        self.fitSpecificBonds = fitSpecificBonds


    def get_param(self, Temp, plot_directory=None, fit_constraints=True):
        """Calculate bond parameters"""

        print('Total bonds:', len(self.bond_dists.items()))
        for name, dists in sorted(self.bond_dists.items()):
            st = sorted([name[1:4], name[6:9]])
            sortedName = '%s-%s' % (st[0], st[1])

            # normalize distance counts by spherical shell volume
            RR, ncounts = self.renorm_bonds(dists, self.bin_edges)
            # Drop zero counts
            RR_nz = RR[ncounts>0]
            ncounts_nz = ncounts[ncounts>0]
            dG_nz = -1*kB*Temp*np.log(ncounts_nz)

            fit_bounds = (-np.inf, np.inf)
        
            # if name in bondsFitHarmonic:
            func = harmonic
            print('=====> Fitting a harmonic for bond ', name)
            popt, _ = curve_fit(func, RR_nz, dG_nz,
                            p0=[3.7, 1.0, -9],# 3.7 1.0 -9
                            bounds=fit_bounds,
                            maxfev=300000)
        
            popt[2] = 0 # set constant offset to zero for the unit test, as forces.py uses a V0=0 offset.

            bond_noise_level = 1  # Noise level            
            
            # Define the Gaussian Process Kernel with controllable parameters
            bond_lengthscale = 0.2  # Lengthscale parameter (RBF)
            bond_signal_variance = 1.0  # Output variance (signal variance)              

            # Define the custom mean function using the fitted polynomial
            def custom_mean(X_input):
                return func(X_input.ravel(), *popt)


            kernel = ConstantKernel(constant_value=bond_signal_variance) * RBF(length_scale=bond_lengthscale) + WhiteKernel(noise_level=bond_noise_level)

            # Initialize the Gaussian Process Regressor
            gp = GPCustom(custom_mean, kernel=kernel, optimizer=None)

            # Reshape data for GP fitting (GPs expect 2D inputs)
            # x = RR_nz.reshape(-1, 1)  # Independent variable
            # y = dG_nz  # Dependent variable
            
            seen_range = [np.min(RR_nz), np.max(RR_nz)]
            print('seen range', seen_range)
            # adsa
            x = np.linspace(*seen_range, 1000).reshape(-1,1)
            y = func(x, *popt).reshape(-1)


            # fit multiple nets with different weights initializations, return the best one
            nrNets = 1 #1-3
            nnets = [NeuralNet(numLayers=6, nrNeurons=20) for n in range(nrNets)]
            optimizers = [optim.Adam(nnet.parameters(), lr=0.001) for nnet in nnets]  # Adam 
            # schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0001) for optimizer in optimizers]
            schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7) for optimizer in optimizers]
            # hardBonds = ['(CAA, CAS)']
            thresh = 0.0004 #if name not in hardBonds else 0.00001
            # epochs=30000 if name not in hardBonds else 70000
            nnets, bestNet, bestIndex = fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='bonds', addRange=0.1, optimizers=optimizers, schedulers=schedulers, xlabel='distance (A)', epochs=60000, thresh=thresh) # 30k-50k epochs, 0.004-0.01 thresh

            results = {'nnets': nnets, 'bestNet': bestNet, 'bestIndex': bestIndex, 'popt' : popt.tolist()}
            
            # pickle the neural nets for this bond
            
            # with open(f'{plot_directory}/bonds-{sortedName}_nnets.pkl', 'wb') as f:
            #     pickle.dump(results, f)


            self.prior_bond[sortedName] = results

        return self.prior_bond


''' like ParamAngleFlexCalculator but the NN is fitted using data generated from the polynomial fit, in order to mimic the classical potential as closely as possible'''
class ParamAngleFlexCalculatorDataFromPoly(ParamAngleCalculator):
    def __init__(self, center=False, fitSpecificAngles=None):
        super().__init__(center)
        self.fitSpecificAngles = fitSpecificAngles

    def get_param(self, Temp, plot_directory=None, fit_constraints=True):
        for name, thetas in self.thetas.items():
            RR, ncounts = renorm_angles(thetas, self.bin_edges)

            # Drop zero counts
            RR_nz = RR[ncounts>0]
            ncounts_nz = ncounts[ncounts>0]
            dG_nz = -kB*Temp*np.log(ncounts_nz)

            if fit_constraints:
                fit_bounds = [[0,0,-np.inf], [np.pi,np.inf,np.inf]]
            else:
                fit_bounds = (-np.inf, np.inf)

            # Angle values are in degrees
            func = harmonic
            popt, _ = curve_fit(func, RR_nz, dG_nz,
                                p0=[np.pi/2, 60, -1],
                                bounds=fit_bounds,
                                maxfev=100000)
            
            popt[2] = 0 # set constant offset to zero for the unit test, as forces.py uses a V0=0 offset.

            # Define the Gaussian Process Kernel with controllable parameters
            angle_lengthscale = 0.3  # Lengthscale parameter (RBF)
            angle_noise_level = 0.01  # Noise level
            angle_signal_variance = 0.2  # Output variance (signal variance)

            customMeanShift = 0            
            # Define the custom mean function using the fitted polynomial
            def custom_mean(X_input):
                return func(X_input.ravel(), *popt) + customMeanShift # add +2 as the extremes are not well captured by the harmonic function

            kernel = ConstantKernel(constant_value=angle_signal_variance) * RBF(length_scale=angle_lengthscale)  + WhiteKernel(noise_level=angle_noise_level)
            # Initialize the Gaussian Process Regressor
            gp = GPCustom(custom_mean, kernel=kernel, optimizer=None)

            # Reshape data for GP fitting (GPs expect 2D inputs)
            # x = RR_nz.reshape(-1, 1)  # Independent variable
            # y = dG_nz  # Dependent variable

            print('seen range', np.min(RR_nz), np.max(RR_nz))
            # adsa
            x = np.linspace(0.8, np.pi, 1000).reshape(-1,1)
            y = func(x, *popt).reshape(-1)
            # print('x', x.shape)
            # print('y', y.shape)

            # fit multiple nets with different weights initializations, return the best one
            nrNets = 1
            nnets = [NeuralNet(numLayers=2, nrNeurons=20) for n in range(nrNets)]
            optimizers = [optim.Adam(nnet.parameters(), lr=0.002) for nnet in nnets]  
            schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7) for optimizer in optimizers]
            thresh = 0.00001 if 'CAW' not in name else 0.00001
            nnets, bestNet, bestIndex = fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='angle', addRange=0.3, optimizers=optimizers, schedulers=schedulers, xlabel='angle (0-pi)', epochs=10000, customMeanShift=customMeanShift, thresh=thresh)

            self.prior_angle[name] = {'nnets': nnets, 'bestNet': bestNet, 'bestIndex': bestIndex, 'popt': popt.tolist()}

        return self.prior_angle




""" This class fits a polynomial to the dihedral data, and then generates data from the polynomial to fit a neural network to it."""
class ParamDihedralCalculatorDataFromPoly(ParamDihedralFlexCalculator):
    def __init__(self, terms=2, unified=False, scale=1):
        super().__init__(terms, unified, scale)
    
    def get_param(self, Temp, plot_directory=None, fit_constraints=True):
        for name, thetas in self.thetas.items():
            # Dihedrals don't require normalization (all the binds are the same size),
            # but we still need to convert to degrees
            RR = .5*(self.bin_edges[1:]+self.bin_edges[:-1]) # bin centers
            ncounts = thetas

            # Drop zero counts
            RR_nz = RR[ncounts>0]
            ncounts_nz = ncounts[ncounts>0]
            dG_nz = -kB*Temp*np.log(ncounts_nz)

            # Fit may fail, better to try-catch. p0 usually not necessary if function is reasonable.
            p0 = [0] # The first parameter is an arbitrary offset from zero
            for i in range(self.n_terms):
                p0.append(0.1)
                p0.append(i/self.n_terms)

            func = self.dihedral_fit_fun
            popt, _ = curve_fit(func, RR_nz, dG_nz, p0=p0, maxfev=100000, xtol=1e-10, ftol=1e-10)

            print('popt', popt)
            # import pdb; pdb.set_trace()

            # remove arbitrary offset in order to match with forces.py
            popt[0] = 0

            # Define the Gaussian Process Kernel with controllable parameters
            lengthscale = 0.1  # Lengthscale parameter (RBF)
            noise_level = 1e-2  # Noise level
            signal_variance = 1.0  # Output variance (signal variance)
            
            # Define the custom mean function using the fitted polynomial
            def custom_mean(X_input):
                return func(X_input.ravel(), *popt)

            kernel = ConstantKernel(constant_value=signal_variance) * RBF(length_scale=lengthscale) + WhiteKernel(noise_level=noise_level)

            # Initialize the Gaussian Process Regressor
            # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            gp = GPCustom(custom_mean, kernel=kernel, n_restarts_optimizer=10)

            # Reshape data for GP fitting (GPs expect 2D inputs)
            # x = RR_nz.reshape(-1, 1)  # Independent variable
            # y = dG_nz  # Dependent variable
            # print('x', x.shape)
            # print('y', y.shape)

            # x = np.linspace(-180, 180, 1000).reshape(-1,1)
            x = np.linspace(-np.pi, np.pi, 1000).reshape(-1,1)
            y = func(x, *popt).reshape(-1)
            # print('x', x.shape)
            # print('y', y.shape)
            
            # add -180 and +180 to xs, and stack them into a 3x bigger array
            # x = np.concatenate([x-180, x, x+180])
            # y = np.concatenate([y, y, y]) # add the same y values to the new xs 

            # fit multiple nets with different weights initializations, return the best one
            nrNets = 1
            nnets = [NeuralNet(numLayers=6, nrNeurons=20) for n in range(nrNets)]
            optimizers = [optim.Adam(nnet.parameters(), lr=0.002) for nnet in nnets]  # Adam 
            #schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=0.0002) for optimizer in optimizers]
            # set a normal annealing without cosine
            schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8) for optimizer in optimizers]
            # nnets, bestNet, bestIndex = fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='dihedral', addRange=0.1, optimizers=optimizers, schedulers=schedulers, epochs=10000, periodicRange=[-180, 180])
            nnets, bestNet, bestIndex = fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='dihedral', addRange=0.1, optimizers=optimizers, schedulers=schedulers, xlabel='dihedral angle (-pi, pi)', epochs=30000, periodicRange=[-np.pi, np.pi], thresh=0.00001)


            # save the neural net parameters
            self.prior_dihedral[name] = {'nnets': nnets, 'bestNet': bestNet, 'bestIndex': bestIndex, 'offset': popt[0].tolist()}
            
        return self.prior_dihedral


class PreprocessorTestDeltaForces(Preprocessor):
    def __init__(self, input_paths, save_path, prior_builder, prior_file, prior_name, num_frames, temp, optimize_forces, box, prior_plots, resume_preprocess, num_cores):
        self.input_paths = input_paths
        self.save_path = save_path
        self.prior_builder = prior_builder
        self.prior_file = prior_file
        self.num_frames = num_frames
        self.temp = temp

        if os.path.exists('pdb_list.pkl'):
            with open('pdb_list.pkl', 'rb') as f:
                self.pdbid_list = pickle.load(f)
        else:
            self.pdbid_list = self.get_pdbid_list()
            
            if FILTER_NOT_PROCESSED_STEP_ONE:
                pdbs_processed_step1 = [f.split('/')[-3] for f in glob.glob(self.save_path + "/*/fit/fit_ok.txt")]
                # print('pdbs_processed', len(pdbs_processed_step1))
                
                # remove keys that are not in pdbs_processed_step1
                self.pdbid_list = {k: v for k, v in self.pdbid_list.items() if k in pdbs_processed_step1}
                print('%d pdbs left after removing pdbs not processed in step 1' % len(self.pdbid_list))

            # pickle pdb_list 
            with open('pdb_list.pkl', 'wb') as f:
                pickle.dump(self.pdbid_list, f)

        # take only 30 pdbs for testing
        # self.pdbid_list = {k: v for k, v in self.pdbid_list.items() if k in list(self.pdbid_list.keys())[:30]}   

        self.optimize_forces = optimize_forces
        self.box = box
        self.prior_plots = prior_plots
        self.resume_preprocess = resume_preprocess
        self.num_cores = num_cores

        print("Input directory path:", self.input_paths)
        print("Save directory path:", self.save_path)
        print(f"Temperature: {self.temp}")
        print("Number of frames:", self.num_frames or "all")
        print("Number of cores used for parallelization:", self.num_cores)
        # print("PDB ID list:", self.pdbid_list)

    def get_pdbid_list(self):
        pdbid_list = dict()

        for input_path in self.input_paths:
            file_names = os.listdir(input_path)
            # print('input_path', input_path)
            # print('file_names', file_names)
            # ?asdas
            for file_name in sorted(file_names):
                # self.pdbid_list = [p for p in self.pdbid_list if os.path.basename(p) in ['1GPQ', '1EI8']]
                # if f'{file_name}' in ['1GPQ', '1EI8']:
                # if f'{file_name}' in ['1GPQ']:
                    if (os.path.exists(os.path.join(input_path, file_name, "result", f"output_{file_name}.h5"))):
                        pdbid_list[f'{file_name}'] = os.path.join(input_path, file_name)
                    else:
                        print(f"  Skipping \"{file_name}\" (directory contains no output)")
        
        print('pdbid_list', pdbid_list)
        return pdbid_list



    def preprocess(self, pdbids=None):
        os.makedirs(os.path.join(self.save_path, "result"), exist_ok=True)

        if not pdbids:
            pdbids = self.pdbid_list
        
        for pdbid in pdbids:
            self.process_step1(pdbid, bar_position=0)
            
        regen = False
        if not os.path.exists(self.save_path + '/prior_builder.pkl') or regen:
        
            # Merge cache files back into prior builder
            for pdbid in tqdm(pdbids, desc="Merging cache files together"):
                cache_dir = os.path.join(self.save_path, pdbid, "fit")
                self.prior_builder.load_molecule_cache(cache_dir)

            with open(self.save_path + '/prior_builder.pkl', 'wb') as f:
                pickle.dump(self.prior_builder, f)
        else:
            with open(self.save_path + '/prior_builder.pkl', 'rb') as f:
                self.prior_builder = pickle.load(f)

        self.process_step2()
        
        self.prior_builder.save_prior(self.save_path, None)  
        
        return self.process_step3(pdbids[0])
        # self.process_step3('3GUQ')
        # asdsa
        
        # with open(os.path.join(self.save_path, "result/ok_list.txt"), "wt", encoding="utf-8") as ok_list:
        #     ok_list.write("\n".join(pdbids))

    
    def process_step3(self, pdbid, bar_position=0):
        """Save prior focefield and generate delta forces data for each protein"""
        output_path = os.path.join(self.save_path, pdbid)

        # Remove legacy prior files if they exist
        if os.path.exists(f"{output_path}/raw/{pdbid}_priors.yaml"):
            os.unlink(f"{output_path}/raw/{pdbid}_priors.yaml")
        if os.path.exists(f"{output_path}/raw/{pdbid}_prior_params.json"):
            os.unlink(f"{output_path}/raw/{pdbid}_prior_params.json")

        # Generate delta forces for all atom simulation vs. prior FF
        coords_npz = f'{output_path}/raw/coordinates.npy'
        box_npz = None
        if self.box:
            box_npz = f"{output_path}/raw/box.npy"
        forcefield = os.path.join(self.save_path, "priors.yaml")
        psf_file = f'{output_path}/processed/{pdbid}_processed.psf'
        prior_params = self.prior_builder.prior_params
        print('prior_params', prior_params)
        device = 'cpu'

        # make_deltaforces(coords_npz, forces_npz, delta_forces_npz, box_npz, forcefield, psf_file,
        #                  prior_params["exclusions"], device, prior_params["forceterms"], bar_position=bar_position)

        deltaForcesObj = DeltaForcesTest(device, psf_file, coords_npz, box_npz)

        if 'external' in self.prior_builder.prior_params.keys():
            # forceterms = ['bonds', 'angles', 'dihedrals']
            otherParams = deltaForcesObj.addExternalForces(forcefield, self.prior_builder.priors['bonds'], self.prior_builder.priors['angles'], self.prior_builder.priors['dihedrals'], forceterms=prior_params["forceterms_nn"], bar_position=bar_position, frames=[0])

            # forceterms = ['repulsioncg'] # update them properly in preprocess.py in the _flex class
            # deltaForcesObj.computePriorForces(forcefield, exclusions=prior_params["exclusions"],
            #     forceterms=prior_params["forceterms_classical"], bar_position=bar_position)
        else:
            # forceterms = ['angles', 'repulsioncg']
            otherParams = deltaForcesObj.computePriorForces(forcefield, exclusions=prior_params["exclusions"],
                forceterms=prior_params["forceterms"], bar_position=bar_position, frames=[0])
        
        # print('keys', self.prior_builder.priors['dihedrals']['(X, X, X, X)'].keys())
        # otherParams[0]['dihedralOffset'] = self.prior_builder.priors['dihedrals']['(X, X, X, X)']['offset']

        return deltaForcesObj.prior_forces, otherParams
        # load MD forces from forces_npz, compute delta forces, and save them in delta_forces_npz
        # deltaForcesObj.makeAndSaveDeltaForces(forces_npz, delta_forces_npz) 




def print_tensor_comparison(name, tensor_flex, tensor_std, printNevertheless=False, atol=1e-6):
    """
    Print and compare two tensors with an optional assertion for similarity.
    """
    if isinstance(tensor_flex, np.ndarray):
        tensor_flex = torch.tensor(tensor_flex)
    if isinstance(tensor_std, np.ndarray):
        tensor_std = torch.tensor(tensor_std)

    assert tensor_flex.shape[0] == tensor_std.shape[0], f"Tensor shapes do not match: tensor_flex {tensor_flex.shape} vs tensor_std {tensor_std.shape}"
    diff = torch.abs(tensor_flex - tensor_std)
    # print('diff.shape', diff.shape)
    # print('discrepancy indices', np.where(diff > 1e-06))
    if torch.sum(diff) < atol:
        print(f"{name} values match within tolerance {atol}\n")
        if printNevertheless:
            print(f"{name} flex:", tensor_flex[:10])
            print(f"{name} std:", tensor_std[:10])
    else:
        print(f"{name} values do NOT match! Difference: {diff[:10]}\n")
        print(f"{name} flex:", tensor_flex[:10])
        print(f"{name} std:", tensor_std[:10])


def cosinesim_comparison(name, tensor_flex, tensor_std, atol=0.99):
    """
    Print and compare two tensors with an optional assertion for similarity.
    """

    assert tensor_flex.shape[0] == tensor_std.shape[0], f"Tensor shapes do not match: {tensor_flex.shape} vs {tensor_std.shape}"
    # if tensors are numpy, convert them to torch tensors
    if isinstance(tensor_flex, np.ndarray):
        tensor_flex = torch.tensor(tensor_flex)
    if isinstance(tensor_std, np.ndarray):
        tensor_std = torch.tensor(tensor_std)

    cosine_sim = torch.nn.functional.cosine_similarity(tensor_flex, tensor_std, dim=0)

    # print('diff.shape', diff.shape)
    # print('discrepancy indices', np.where(diff > 1e-06))
    if cosine_sim > atol:
        print(f"{name} values are almost the same! Cosine similarity of {cosine_sim}\n")
        # if True:
        #     print(f"{name} flex:", tensor_flex[:10])
        #     print(f"{name} std:", tensor_std[:10])
    else:
        print(f"{name} values do NOT match! Cosine similarity: {cosine_sim}\n")
        print(f"{name} flex:", tensor_flex[:10])
        print(f"{name} std:", tensor_std[:10])



# class TestMatch(unittest.TestCase):
#     def test_match(self, r12):


class DeltaForcesTest:
    def __init__(self, device, psf, coords_npz, box_npz):
        self.device = torch.device(device)
        self.precision = torch.float32
        self.replicas = 1

        self.mol = Molecule(psf)
        self.natoms = self.mol.numAtoms

        self.coords = np.load(coords_npz)
        self.box = None
        if box_npz:
            self.box = np.load(box_npz)
        
        self.coords = torch.tensor(self.coords, dtype=self.precision).to(device)

        self.atom_vel = torch.zeros(self.replicas, self.natoms, 3)
        self.atom_pos = torch.zeros(self.natoms, 3, self.replicas)
        if self.box is not None:
            # Reshape box to be rectangle, then format to be given to set_box
            linearized = self.box.reshape(-1,9).take([0,4,8],axis=1)
            self.box_full = linearized.reshape(linearized.shape[0], 3, 1)
        else:
            self.box_full = torch.zeros(self.coords.shape[0], 3, 1)

        
        self.prior_forces = torch.zeros((self.coords.shape[0], self.natoms, 3), dtype=self.precision).to('cpu') # store these on CPU
        self.parameters = None
        

    def computePriorForces(self,
        forcefield,
        exclusions=("bonds"),
        forceterms=["Bonds", "Angles", "RepulsionCG"],
        bar_position=0,frames=None
    ):

        print('forcefield', forcefield)
        # if forceterms is empty list, then we exit
        if forceterms == []:
            return

        ff = tagged_forcefield.create(self.mol, forcefield)
        parameters = Parameters(ff, self.mol, forceterms, precision=self.precision, device=self.device)

        system = System(self.natoms, self.replicas, self.precision, self.device)
        system.set_positions(self.atom_pos)
        system.set_velocities(self.atom_vel)

        forces = Forces(parameters, terms=forceterms, exclusions=exclusions)
        otherparams = [i for i in range(0, self.coords.shape[0])]
        if frames is None: # if None, then process all frames
            frames = range(0, self.coords.shape[0])
        
        start_time = time.time()
        for i in tqdm(frames, position=bar_position, dynamic_ncols=True, desc="Delta forces - Classical", leave=(bar_position==0)):
            co = self.coords[i]
            system.set_box(self.box_full[i])
            _, otherparams[i] = forces.compute(co.reshape([1, self.natoms, 3]), system.box, system.forces)
            fr = (
                system.forces.detach().cpu().reshape([self.natoms, 3])
            )
            self.prior_forces[i,:,:] += fr
        print('Time taken for classical forces', time.time() - start_time)
        return otherparams

    def makeAndSaveDeltaForces(self, forces_npz, delta_forces_npz):
        all_forces = np.load(forces_npz)
        prior_forces_npy = np.array(self.prior_forces.detach().cpu())
        delta_forces = all_forces - prior_forces_npy
        np.save(delta_forces_npz, delta_forces)

    def addExternalForces(self, forcefield, nnetsBonds, nnetsAngles, nnetsDihedrals, forceterms, bar_position=0, frames=None):

        # if forceterms is empty list, then we exit
        if forceterms == []:
            return
       
        parameters = ParametersNN(self.mol, forceterms, precision=self.precision, device=self.device)

        # for adding the neural network priors. ExternalNN is molecule-agnostic
        calc = ExternalNN(parameters, nnetsBonds, nnetsAngles, nnetsDihedrals, forceterms, self.device)
        tensorbox = torch.tensor(self.box, dtype=self.precision).to(self.device)

        if frames is None: # if None, then process all frames
            frames = range(0, self.coords.shape[0])
        

        start_time = time.time()
        _, forces, otherparams = calc.calculate(self.coords, tensorbox)
        self.prior_forces += forces.detach().cpu()
        print('Time taken for neural network forces', time.time() - start_time)

        return otherparams  

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("input", nargs = "+", help="Input directory path")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument("--pdbids", nargs="*", help="List of specific PDB IDs to process")
    parser.add_argument("--num-frames", "--num_frames", type=int, default=None, help="Number of frames to process")
    parser.add_argument("--temp", type=int, default=300, help="Temperature")
    parser.add_argument("--optimize-forces", action="store_true", help="Use statistically optimal force aggregation (Kramer 2023)")
    parser.add_argument("--prior-file", default=None, help="Use PRIOR_FILE instead of fitting a prior")
    parser.add_argument('--no-box', default=False, action='store_true', help="Don't use periodic box information")
    parser.add_argument('--prior-plots', default=True, action='store_true', help="Save plots of the prior fit functions")
    parser.add_argument('--no-prior-plots', dest='prior_plots', action='store_false', help="Don't save plots of the prior fit functions")
    parser.add_argument('--no-fit-constraints', default=False, action='store_true', help="Disable range constraints when fitting prior functions")
    parser.add_argument('--tag-beta-turns', default=False, action='store_true', help="Give beta turns a different bond type in the prior")
    parser.add_argument('--resume', default=False, action='store_true', help="Resume processing rather than overwriting, all settings must be identical between calls")
    parser.add_argument('--num-cores', type=int, default=32, help="Number of cores to be used for parallelization of preprocessing")

    args = parser.parse_args()
    print(args)

    input_dirs = args.input
    output_dir = args.output
    pdbids = args.pdbids
    num_frames = args.num_frames
    temp = args.temp
    optimize_forces = args.optimize_forces
    box = not args.no_box
    prior_plots = args.prior_plots
    # prior_name = args.prior
    prior_file = args.prior_file
    resume_preprocess = args.resume
    num_cores = args.num_cores

    
    bonds = False
    if bonds:
        prior_name = 'Prior_CA_lj_angleXCX_dihedralX'
        print(f"Using prior: {prior_name}")
        prior_builder_std = Prior_CA_lj_angleXCX_dihedralX_specific(          forceterms=['bonds']) 
        prior_builder_std.enable_fit_constraints(not args.no_fit_constraints)
        prior_builder_std.enable_bond_tags(args.tag_beta_turns)
        preprocessor_std = PreprocessorTestDeltaForces(input_dirs, output_dir + '_std_bond', prior_builder_std, prior_file, prior_name, num_frames, temp, optimize_forces, box, prior_plots, resume_preprocess, num_cores)
        prior_forces_std, other_params_std = preprocessor_std.preprocess(pdbids)

        prior_name = 'CA_lj_angleXCX_dihedralX_flex'
        print(f"Using prior: {prior_name}")
        prior_builderFlex = Prior_CA_lj_angleXCX_dihedralX_flex_specific(forceterms_nn=['bonds']) 
        prior_builderFlex.enable_fit_constraints(not args.no_fit_constraints)
        prior_builderFlex.enable_bond_tags(args.tag_beta_turns)
        preprocessor_flex = PreprocessorTestDeltaForces(input_dirs, output_dir + '_flex_bond', prior_builderFlex, prior_file, prior_name, num_frames, temp, optimize_forces, box, prior_plots, resume_preprocess, num_cores)
        prior_forces_flex, other_params_flex = preprocessor_flex.preprocess(pdbids)

        print('other_params_flex', other_params_flex.keys())
        print('other_params_std', other_params_std[0].keys())

        print_tensor_comparison("bond_dist", other_params_flex['bond_dist'][0], other_params_std[0]['bond_dist'])
        print('keys', preprocessor_flex.prior_builder.priors['bonds'].keys())

        cosinesim_comparison("Bond Energy", other_params_flex['Eb'][0].reshape(-1), other_params_std[0]['Eb'].reshape(-1))
        cosinesim_comparison("Prior forces along X", prior_forces_flex[0][:,0], prior_forces_std[0][:,0])
        cosinesim_comparison("Prior forces along Y", prior_forces_flex[0][:,1], prior_forces_std[0][:,1])
        cosinesim_comparison("Prior forces along Z", prior_forces_flex[0][:,2], prior_forces_std[0][:,2])


    angles = False
    if angles:
        prior_name = 'Prior_CA_lj_angleXCX_dihedralX'
        print(f"Using prior: {prior_name}")
        prior_builder_std = Prior_CA_lj_angleXCX_dihedralX_specific(forceterms=['angles']) 
        prior_builder_std.enable_fit_constraints(not args.no_fit_constraints)
        prior_builder_std.enable_bond_tags(args.tag_beta_turns)
        preprocessor_std = PreprocessorTestDeltaForces(input_dirs, output_dir + '_std_angles', prior_builder_std, prior_file, prior_name, num_frames, temp, optimize_forces, box, prior_plots, resume_preprocess, num_cores)
        prior_forces_std, other_params_std = preprocessor_std.preprocess(pdbids)

        prior_name = 'CA_lj_angleXCX_dihedralX_flex'
        print(f"Using prior: {prior_name}")
        prior_builderFlex = Prior_CA_lj_angleXCX_dihedralX_flex_specific(forceterms_nn=['angles']) 
        prior_builderFlex.enable_fit_constraints(not args.no_fit_constraints)
        prior_builderFlex.enable_bond_tags(args.tag_beta_turns)
        preprocessor_flex = PreprocessorTestDeltaForces(input_dirs, output_dir + '_flex_angles', prior_builderFlex, prior_file, prior_name, num_frames, temp, optimize_forces, box, prior_plots, resume_preprocess, num_cores)
        prior_forces_flex, other_params_flex = preprocessor_flex.preprocess(pdbids)

        print('other_params_flex', other_params_flex.keys())
        print('other_params_std', other_params_std[0].keys())

        print_tensor_comparison("ra21", other_params_flex['ra21'][0], other_params_std[0]['ra21'])
        print_tensor_comparison("ra23", other_params_flex['ra23'][0], other_params_std[0]['ra23'])
        print_tensor_comparison("theta", other_params_flex['theta'][0].reshape(-1), other_params_std[0]['theta'].reshape(-1), printNevertheless=False, atol=0.001)
        print('keys', preprocessor_flex.prior_builder.priors['angles'].keys())
        # print('keys', preprocessor_flex.prior_builder.priors['angles']['(X, CAR, X)'].keys())       
        anglepar_flex = [preprocessor_flex.prior_builder.priors['angles'][ang]['popt'] for ang in preprocessor_flex.prior_builder.priors['angles'].keys()]
        anglepar_std = other_params_std[0]['angle_params']
        # print('anglepar_flex', anglepar_flex)
        # print('anglepar_std', anglepar_std)
        # print_tensor_comparison("angle parameters", anglepar_flex, anglepar_std)

        # dihedral_offset = other_params_flex[0]['dihedralOffset']
        # print("Using dihedral offset", dihedral_offset)
        cosinesim_comparison("angle E", other_params_flex['Ea'][0].reshape(-1), other_params_std[0]['Ea'].reshape(-1))
        cosinesim_comparison("Prior forces along X", prior_forces_flex[0][:,0], prior_forces_std[0][:,0])
        cosinesim_comparison("Prior forces along Y", prior_forces_flex[0][:,1], prior_forces_std[0][:,1])
        cosinesim_comparison("Prior forces along Z", prior_forces_flex[0][:,2], prior_forces_std[0][:,2])

    dihedrals = True
    if dihedrals:
        prior_name = 'Prior_CA_lj_angleXCX_dihedralX'
        print(f"Using prior: {prior_name}")
        prior_builder_std = Prior_CA_lj_angleXCX_dihedralX_specific(forceterms=['dihedrals'])
        prior_builder_std.enable_fit_constraints(not args.no_fit_constraints)
        prior_builder_std.enable_bond_tags(args.tag_beta_turns)
        preprocessor_std = PreprocessorTestDeltaForces(input_dirs, output_dir + '_std_dihedrals', prior_builder_std, prior_file, prior_name, num_frames, temp, optimize_forces, box, prior_plots, resume_preprocess, num_cores)
        prior_forces_std, other_params_std = preprocessor_std.preprocess(pdbids)

        prior_name = 'CA_lj_angleXCX_dihedralX_flex'
        print(f"Using prior: {prior_name}")
        prior_builderFlex = Prior_CA_lj_angleXCX_dihedralX_flex_specific(forceterms_nn=['dihedrals']) 
        prior_builderFlex.enable_fit_constraints(not args.no_fit_constraints)
        prior_builderFlex.enable_bond_tags(args.tag_beta_turns)
        preprocessor_flex = PreprocessorTestDeltaForces(input_dirs, output_dir + '_flex_dihedrals', prior_builderFlex, prior_file, prior_name, num_frames, temp, optimize_forces, box, prior_plots, resume_preprocess, num_cores)
        prior_forces_flex, other_params_flex = preprocessor_flex.preprocess(pdbids)

        print_tensor_comparison("r12", other_params_flex['r12'][0], other_params_std[0]['r12'])
        print_tensor_comparison("r23", other_params_flex['r23'][0], other_params_std[0]['r23'])
        print_tensor_comparison("r34", other_params_flex['r34'][0], other_params_std[0]['r34'])
        print_tensor_comparison("phi", other_params_flex['phi'][0].reshape(-1), other_params_std[0]['phi'].reshape(-1), atol=0.001)
        # dihedral_offset = other_params_flex[0]['dihedralOffset']
        dihedral_offset = preprocessor_flex.prior_builder.priors['dihedrals']['(X, X, X, X)']['offset']
        print("Using dihedral offset", dihedral_offset)
        cosinesim_comparison("E", other_params_flex['E'][0].reshape(-1) + dihedral_offset, other_params_std[0]['E'].reshape(-1))
        cosinesim_comparison("Prior forces along X", prior_forces_flex[0][:,0], prior_forces_std[0][:,0])
        cosinesim_comparison("Prior forces along Y", prior_forces_flex[0][:,1], prior_forces_std[0][:,1])
        cosinesim_comparison("Prior forces along Z", prior_forces_flex[0][:,2], prior_forces_std[0][:,2])
