import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import pickle
from module.prior import *

sns.set_style("whitegrid")


# original code https://github.com/torchmd/torchmd-cg/blob/master/torchmd_cg/utils/prior_fit.py


def makePlot(xlabel, name, xsPoints, ysPoints, points1, points2, plot_directory, plotType, periodicRange=None):
    print('Plotting ', f'{plotType}-{name}-fit.png')
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(15, 7.5))
    fig.suptitle(name)
    
    def plotax(ax, xsPoints, ysPoints, points, periodicRange):
        ax.plot(xsPoints, ysPoints, 'o')
        for xs, ys, ysStd, label in points:
            ax.plot(xs, ys, label=label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('dG (kcal/mol)')
            # ax[0, 0].set_title(plot_name)
            if ysStd is not None:
                ax.fill_between(
                xs.ravel(),
                ys - 1.96 * ysStd,
                ys + 1.96 * ysStd,
                alpha=0.2)
        ax.legend()
        if periodicRange is not None:
            # plot dashed vertical lines at the periodic boundary
            ax.axvline(x=periodicRange[0], linestyle='--', color='black')
            ax.axvline(x=periodicRange[1], linestyle='--', color='black')

    plotax(axes[0,0], xsPoints, ysPoints, points1, periodicRange)
    plotax(axes[0,1], xsPoints, ysPoints, points2, periodicRange)

    fig.savefig(os.path.join(plot_directory, f'{plotType}-{name}-fit.png'))
    plt.close() # Don't leak the old one


# use e^2 to make the function to go +inf in both directions
def polynomial_fourth_order(x,a,b,c,d,e):
    return a + b*x + c*x**2 + d*x**3 + e**2 * x**4

def polynomial_sixth_order(x,a,b,c,d,e,f,g):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6



# Define the neural network model with dynamic hidden layers
class NeuralNet(nn.Module):
    def __init__(self, numLayers=6, nrNeurons=20):
        super(NeuralNet, self).__init__()

        # Ensure numLayers is at least 1
        if numLayers < 1:
            raise ValueError("numLayers must be at least 1")

        self.numLayers = numLayers
        self.nrNeurons = nrNeurons

        # Define input layer
        self.input_layer = nn.Linear(1, nrNeurons)

        # Define hidden layers dynamically
        self.hidden_layers = nn.ModuleList([
            nn.Linear(nrNeurons, nrNeurons) for _ in range(numLayers)
        ])

        # Define output layer
        self.output_layer = nn.Linear(nrNeurons, 1)

        # Activation function
        self.act = nn.SiLU()  # Swish activation

        # Initialize weights and biases
        self.initialize_weights()

    def forward(self, input):
        # Input layer
        x = self.act(self.input_layer(input))

        # Pass through each hidden layer
        for layer in self.hidden_layers:
            x = self.act(layer(x))

        # Output layer
        return self.output_layer(x)

    def initialize_weights(self):
        # Initialize input layer
        init.uniform_(self.input_layer.weight, a=-0.5, b=0.5)
        init.constant_(self.input_layer.bias, 0)

        # Initialize hidden layers
        for layer in self.hidden_layers:
            init.uniform_(layer.weight, a=-0.5, b=0.5)
            init.constant_(layer.bias, 0)

        # Initialize output layer
        init.uniform_(self.output_layer.weight, a=-0.5, b=0.5)
        init.constant_(self.output_layer.bias, 0)


class GPCustom:
    def __init__(self, custom_mean, *args, **kwargs):
        self.custom_mean = custom_mean
        self.gp = GaussianProcessRegressor(*args, **kwargs)
        # # super(GPCustom, self).__init__(*args, **kwargs)
        # self.scaler_X = StandardScaler()
        # self.scaler_y = StandardScaler()

    def fit(self, x, y):
        y_adjusted = y - self.custom_mean(x)
        self.gp.fit(x, y_adjusted)

        # # Save original X for custom mean calculations
        # self.original_X = x.copy()

        # # Standardize X
        # X_scaled = self.scaler_X.fit_transform(x)

        # # Compute the custom mean on the original (unstandardized) X
        # y_adjusted = y - self.custom_mean(x)

        # # Standardize y after mean adjustment
        # y_scaled = self.scaler_y.fit_transform(y_adjusted.reshape(-1, 1)).ravel()

        # # Fit the Gaussian Process on standardized data
        # self.gp.fit(X_scaled, y_scaled)

    def predict(self, X):
        # y_pred = super(GPCustom, self).predict(X, return_std)
        y_pred, y_std = self.gp.predict(X, return_std=True)
        y_pred += self.custom_mean(X)
        return y_pred, y_std

        # # Standardize X for prediction
        # X_scaled = self.scaler_X.transform(X)

        # # Predict using standardized data
        # y_scaled_pred, y_std_scaled = self.gp.predict(X_scaled, return_std=True)

        # # Inverse-transform the predictions
        # y_pred = self.scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()

        # # Add back the custom mean (computed on original X)
        # y_pred += self.custom_mean(X)

        # # Standard deviation does not change with mean adjustment
        # y_std = y_std_scaled * self.scaler_y.scale_[0]

        # return y_pred, y_std

    
    def predictCustomMean(self, X):
        return self.custom_mean(X)
    

# fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='bonds', addRange=0.5, optimizers=optimizers, schedulers=schedulers, epochs=10000)
def fitGPNN(gp, nnets, x, y, name, plot_directory, plotType, addRange, optimizers, schedulers, xlabel, epochs=10000, mseLoss = nn.MSELoss(), 
    periodicRange=None, lossPeriodWeight=0.01, lossDerivWeight=0.1, customMeanShift=0, thresh=0.001):

    gp.fit(x, y)

    # Generate synthetic data with the GP model outside the range of the original data
    # asds
    seen_range = [np.min(x), np.max(x)]
    xs_range = [seen_range[0] - addRange*(seen_range[1]-seen_range[0]), seen_range[1] + addRange*(seen_range[1]-seen_range[0])]
    # xsnn = np.linspace(1.5, seen_range[1]+4, 100)
    xsnn = np.linspace(*xs_range, 1000) #pyright: ignore[reportCallIssue]
    ysgp = gp.predict(xsnn.reshape(-1, 1))[0]

    # Convert data to PyTorch tensors
    x_train = torch.tensor(xsnn, requires_grad=False, dtype=torch.float32).reshape(-1, 1)
    y_train = torch.tensor(ysgp, requires_grad=False, dtype=torch.float32).reshape(-1, 1)

    finalLosses = [_ for _ in range(len(nnets))]
    for n, (nnet, optimizer, scheduler) in enumerate(zip(nnets, optimizers, schedulers)):   
        # Training loop
        for epoch in range(epochs):
            # Standard MSE loss
            y_pred = nnet(x_train)
            loss = mseLoss(y_pred, y_train)

            if periodicRange is not None:
                # the function should be periodict in the range given by periodicRange. penalize if the loss is not periodic at the edges and if the derivatives are not equal.
                xLower = torch.tensor(periodicRange[0], requires_grad=True, dtype=torch.float32).reshape(-1, 1)
                xUpper = torch.tensor(periodicRange[1], requires_grad=True, dtype=torch.float32).reshape(-1, 1)
                yLower = nnet(xLower)
                yUpper = nnet(xUpper)
                lossPeriod = x_train.shape[0] * lossPeriodWeight * mseLoss(yLower, yUpper)
                
                # compute the derivaties w.r.t. xLower and xUpper with autograd, make sure they won't influence the main backpropagation
                dydxLower = torch.autograd.grad(yLower, xLower, create_graph=True)[0]
                dydxUpper = torch.autograd.grad(yUpper, xUpper, create_graph=True)[0]
                lossDeriv = x_train.shape[0] * lossDerivWeight * mseLoss(dydxLower, dydxUpper)
                loss += lossPeriod + lossDeriv
            
            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.7f}")
                if periodicRange is not None:
                    print(f"Loss Periodic: {lossPeriod.item():.7f}, Loss Deriv: {lossDeriv.item():.7f}") #pyright: ignore[reportPossiblyUnboundVariable]

            # Backward pass and optimization
            optimizer.zero_grad()
            if loss < thresh:# and lossPeriod < 0.001 and lossDeriv < 0.002:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
                if periodicRange is not None:
                    print(f"Loss Periodic: {lossPeriod.item():.4f}, Loss Deriv: {lossDeriv.item():.4f}") #pyright: ignore[reportPossiblyUnboundVariable]
                print('Stopping training early due to convergence')
                break

            loss.backward()
            optimizer.step()
            scheduler.step()

        finalLosses[n] = loss.item() #pyright: ignore[reportPossiblyUnboundVariable]
        plotGPNN(xlabel, plot_directory, seen_range, gp, nnet, plotType, name + '_%d' % n, x, y, periodicRange, customMeanShift)

    bestIndex = np.argmin(finalLosses)
    bestNet = nnets[bestIndex] 

    plotGPNN(xlabel, plot_directory, seen_range, gp, bestNet, plotType, name + '_best', x, y, periodicRange)
            # asda
    return nnets, bestNet, bestIndex

def plotGPNN(xlabel, plot_directory, seen_range, gp, net, plotType, name, x, y, periodicRange, customMeanShift=0):  
    if plot_directory:
        addRangePlot = 0.05
        plot_range = [seen_range[0] - addRangePlot*(seen_range[1]-seen_range[0]), seen_range[1] + addRangePlot*(seen_range[1]-seen_range[0])]
        xs1 = np.linspace(*np.array(plot_range), 100) #pyright: ignore[reportCallIssue]
        # ax[0, 0].plot(plot_space, func(plot_space, *popt))
        ys0 = gp.predictCustomMean(xs1.reshape(-1, 1)) - customMeanShift
        ys1, ys1Std = gp.predict(xs1.reshape(-1, 1))
        ys1nn = net(torch.tensor(xs1, dtype=torch.float32).reshape(-1, 1)).detach().numpy()

        if periodicRange is None:
            addRange2 = 1
            plot_range2 = [seen_range[0] - addRange2*(seen_range[1]-seen_range[0]), seen_range[1] + addRange2*(seen_range[1]-seen_range[0])]

            xs2 = np.linspace(*np.array(plot_range2), 100) #pyright: ignore[reportCallIssue]
            ys2, ys2Std = gp.predict(xs2.reshape(-1, 1))
            ys2nn = net(torch.tensor(xs2, dtype=torch.float32).reshape(-1, 1)).detach().numpy()
        else:
            # put the periodic range as the xs
            xs2 = np.linspace(*np.array(seen_range), 100) #pyright: ignore[reportCallIssue]
            ys2, ys2Std = gp.predict(xs2.reshape(-1, 1))
            ys2nn = net(torch.tensor(xs2, dtype=torch.float32).reshape(-1, 1)).detach().numpy()

            # to the xs2, stack 10% of the xs on either side to show the discontinuity (if any) at the periodic boundary. 
            periodSize = periodicRange[1] - periodicRange[0]
            xs2 = np.concatenate([xs2[-10:] - periodSize, xs2, xs2[:10] + periodSize])
            ys2 = np.concatenate([ys2[-10:], ys2, ys2[:10]]) # Ignore the GP discontinuity, did it so the plot functions wouldn't error out to wrong dimensions
            ys2Std = np.concatenate([ys2Std[-10:], ys2Std, ys2Std[:10]])
            ys2nn = np.concatenate([ys2nn[-10:], ys2nn, ys2nn[:10]])


        points1 = [(xs1, ys1, ys1Std, 'GP'), (xs1, ys1nn, None, 'NN'), (xs1, ys0, None, 'Poly')] # points for left plot with the restricted xs
        points2 = [(xs2, ys2, ys2Std, 'GP'), (xs2, ys2nn, None, 'NN')] # points for left plot with the extended xs

        makePlot(xlabel, name, x, y, points1, points2, plot_directory, plotType, periodicRange)


"""
This performs the following steps: 
1) Poly: fits a 4th order polynomial to the data that extrapolates well outside the data range
2) GP: fits a Gaussian Process to the residual from the 4th order polynomial (this is because it's a very smooth model and automatically tends to zero outside the extrapolation range)
3) Distillation with NN: fits a Neural Network on synthetic GP data that includes xs outside the data range. Given the GP extrapolated well, the NN will too, at least up within the synthetic data range"""
class ParamBondedFlexCalculator(ParamBondedCalculator):
    # if you only want to fit [(CAG, CAH), (CAG, CAL)] use fitSpecificBonds. If None, fit all
    def __init__(self, unified=False, fitSpecificBonds=None):
        super().__init__(unified)
        self.fitSpecificBonds = fitSpecificBonds

    def merge_hists(self, hists):
        bondList = list(hists.keys()) if self.fitSpecificBonds is None else self.fitSpecificBonds
        # print('bondList', bondList)

        for bond in bondList:
            if bond not in self.bond_dists:
                self.bond_dists[bond] = np.zeros(self.num_bins)
            self.bond_dists[bond] += hists[bond]

    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0): #pyright: ignore[reportIncompatibleMethodOverride]
        """Calculate bond parameters"""

        # TODO: double check (CAL, CAM) as its funny, has a flat shape on the left. Daniel says we'll need to hunt down the trajectories and look at that. 
        bondsFitHarmonic = ['(CAR, CAY)', '(CAW, CAY)', '(CAC, CAD)', '(CAC, CAF)', '(CAC, CAR)', '(CAC, CAT)', '(CAF, CAW)', '(CAI, CAW)', '(CAL, CAM)'] # fit harmonic potential for these bonds
        # refit = ['(CAD, CAK)', '(CAD, CAN)', '(CAD, CAQ)', '(CAD, CAS)', '(CAD, CAT)', '(CAD, CAY)', '(CAE, CAG)', '(CAE, CAI)', '(CAE, CAL)', '(CAE, CAM)', '(CAE, CAN)', '(CAF, CAV)', '(CAG, CAN)', '(CAG, CAS)', '(CAG, CAT)', '(CAH, CAG)', '(CAH, CAK)', '(CAH, CAW)', '(CAI, CAY)', '(CAK, CAL)', '(CAK, CAT)', '(CAK, CAW)', '(CAL, CAP)', '(CAM, CAN)', '(CAM, CAP)', '(CAM, CAS)', '()']
        refit = []

        # todosSorted = sorted(refit)
        todosSorted = [] + refit # add individual bonds here that you need to refit
        print('Total bonds:', len(self.bond_dists.items()))
        for name, dists in sorted(self.bond_dists.items()):
            # since (CAT, CAL) and (CAL, CAT) are the same, sort alphabetically to have unique names. Also change from (CAT, CAL) to CAT-CAL as it's more compact
            # print('sortedName', sortedName)
            st = sorted([name[1:4], name[6:9]])
            sortedName = '%s-%s' % (st[0], st[1])

            if os.path.exists(f'{plot_directory}/bonds-{sortedName}_nnets.pkl') and name not in todosSorted:
                print('Founds results for ', name)
                with open(f'{plot_directory}/bonds-{sortedName}_nnets.pkl', 'rb') as f:
                    results = pickle.load(f)
            else:
                # normalize distance counts by spherical shell volume
                RR, ncounts = self.renorm_bonds(dists, self.bin_edges)
                # Drop zero counts
                RR_nz = RR[ncounts>0]
                ncounts_nz = ncounts[ncounts>0]
                dG_nz = -1*kB*Temp*np.log(ncounts_nz)

                fit_bounds = (-np.inf, np.inf)
            
                if name in bondsFitHarmonic:
                    func = harmonic
                    print('=====> Fitting a harmonic for bond ', name)
                    popt, _ = curve_fit(func, RR_nz, dG_nz,
                                    p0=[3.7, 1.0, -9],# 3.7 1.0 -9
                                    bounds=fit_bounds,
                                    maxfev=300000)
                
                    # since these bonds have outliers, use a higher noise level
                    bond_noise_level = 1  # Noise level            
                else:
                    func = polynomial_fourth_order
                    popt, _ = curve_fit(func, RR_nz, dG_nz,
                                    p0=[1, 0.0, 1, 0, 1],
                                    bounds=fit_bounds,
                                    maxfev=300000)

                    bond_noise_level = 1e-2  # Noise level

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
                x = RR_nz.reshape(-1, 1)  # Independent variable
                y = dG_nz  # Dependent variable

                # fit multiple nets with different weights initializations, return the best one
                nrNets = 1 #1-3
                nnets = [NeuralNet(numLayers=6, nrNeurons=20) for _ in range(nrNets)]
                optimizers = [optim.Adam(nnet.parameters(), lr=0.001) for nnet in nnets]  # Adam 
                # schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0001) for optimizer in optimizers]
                schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.7) for optimizer in optimizers]
                nnets, bestNet, bestIndex = fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='bonds', addRange=0.5, optimizers=optimizers, schedulers=schedulers, xlabel='distance (A)', epochs=3000, thresh=0.04) # 30k-50k epochs, 0.004-0.01 thresh

                results = {'nnets': nnets, 'bestNet': bestNet, 'bestIndex': bestIndex}
                
                # pickle the neural nets for this bond
                
                with open(f'{plot_directory}/bonds-{sortedName}_nnets.pkl', 'wb') as f:
                    pickle.dump(results, f)


            self.prior_bond[sortedName] = results

        return self.prior_bond


class ParamAngleFlexCalculator(ParamAngleCalculator):
    def __init__(self, center=False, fitSpecificAngles=None):
        super().__init__(center)
        self.fitSpecificAngles = fitSpecificAngles

    def merge_hists(self, hists):
        angleList = list(hists.keys()) if self.fitSpecificAngles is None else self.fitSpecificAngles
        for angle in angleList:
            # Calculate the value (in radians) for each angle in the prior
            if angle not in self.thetas:
                self.thetas[angle] = np.zeros(self.num_bins)
            self.thetas[angle] += hists[angle]

    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0): #pyright: ignore[reportIncompatibleMethodOverride]
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

            # Define the Gaussian Process Kernel with controllable parameters
            angle_lengthscale = 0.3  # Lengthscale parameter (RBF)
            angle_noise_level = 0.01  # Noise level
            angle_signal_variance = 0.2  # Output variance (signal variance)

            customMeanShift = 2            
            # Define the custom mean function using the fitted polynomial
            def custom_mean(X_input):
                return func(X_input.ravel(), *popt) + customMeanShift # add +2 as the extremes are not well captured by the harmonic function

            kernel = ConstantKernel(constant_value=angle_signal_variance) * RBF(length_scale=angle_lengthscale)  + WhiteKernel(noise_level=angle_noise_level)
            # Initialize the Gaussian Process Regressor
            gp = GPCustom(custom_mean, kernel=kernel, optimizer=None)

            # Reshape data for GP fitting (GPs expect 2D inputs)
            x = RR_nz.reshape(-1, 1)  # Independent variable
            y = dG_nz  # Dependent variable
            
            # fit multiple nets with different weights initializations, return the best one
            nrNets = 2
            nnets = [NeuralNet(numLayers=2, nrNeurons=20) for _ in range(nrNets)]
            optimizers = [optim.Adam(nnet.parameters(), lr=0.002) for nnet in nnets]  
            schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7) for optimizer in optimizers]
            # nnets, bestNet, bestIndex = fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='dihedral', addRange=0.1, optimizers=optimizers, schedulers=schedulers, epochs=10000, periodicRange=[-180, 180])
            nnets, bestNet, bestIndex = fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='angle', addRange=0.5, optimizers=optimizers, schedulers=schedulers, xlabel='angle (0-pi)', epochs=30000, customMeanShift=customMeanShift)

            self.prior_angle[name] = {'nnets': nnets, 'bestNet': bestNet, 'bestIndex': bestIndex}

        return self.prior_angle



class ParamDihedralFlexCalculator(ParamDihedralCalculator):
    def __init__(self, terms=2, unified=False, scale=1):
        super().__init__(terms, unified=unified, scale=scale)
    
    # # like the parent but in radians
    # def dihedral_fit_fun(self, theta, offset, *args):
    #     # Implements the TorchMD torsion function
    #     # https://doi.org/10.1021/acs.jctc.0c01343?rel=cite-as&ref=PDF&jav=VoR
    #     # args = [phi_k0, phase0, phi_k1, phase1, ...]
    #     assert len(args) == self.n_terms*2
    #     result = offset
    #     for i in range(0,self.n_terms):
    #         phi_k = args[i*2]
    #         phase = args[i*2+1]
    #         per = i+1
    #         result += phi_k*(1+np.cos(per*theta - phase) )
    #     return result

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
            result += phi_k*(1+np.cos(per*theta - phase))
        return result
    
    # def dihedral_fit_fun(self, theta, offset, *args):
    #     # Implements the TorchMD torsion function
    #     # https://doi.org/10.1021/acs.jctc.0c01343?rel=cite-as&ref=PDF&jav=VoR
    #     # args = [phi_k0, phase0, phi_k1, phase1, ...]
    #     assert len(args) == self.n_terms*2
    #     result = offset
    #     for i in range(0,self.n_terms):
    #         phi_k = args[i*3]
    #         phase = args[i*3+1]
    #         per = args[i*3+2]
    #         result += phi_k*(1+np.cos(per*theta - phase))
    #     return result

    def get_param(self, Temp, plot_directory=None, fit_constraints=True, min_cnt=0): #pyright: ignore[reportIncompatibleMethodOverride]
        for name, thetas in self.thetas.items():
            # Dihedrals don't require normalization (all the binds are the same size),
            # but we still need to convert to degrees
            # RR = .5*(self.bin_edges[1:]+self.bin_edges[:-1])*180/np.pi  # bin centers
            RR = .5*(self.bin_edges[1:]+self.bin_edges[:-1])  # bin centers
            ncounts = thetas

            # Drop zero counts
            RR_nz = RR[ncounts>0]
            ncounts_nz = ncounts[ncounts>0]
            dG_nz = -kB*Temp*np.log(ncounts_nz)

            # Fit may fail, better to try-catch. p0 usually not necessary if function is reasonable.
            p0: list[float] = [0] # The first parameter is an arbitrary offset from zero
            for i in range(self.n_terms):
                p0.append(0.1)
                p0.append(i/self.n_terms)

            func = self.dihedral_fit_fun
            popt, _ = curve_fit(func, RR_nz, dG_nz, p0=p0, maxfev=100000)

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
            x = RR_nz.reshape(-1, 1)  # Independent variable
            y = dG_nz  # Dependent variable
            # print('x', x.shape)
            # print('y', y.shape)

            # add -180 and +180 to xs, and stack them into a 3x bigger array
            # x = np.concatenate([x-180, x, x+180])
            # y = np.concatenate([y, y, y]) # add the same y values to the new xs 

            # fit multiple nets with different weights initializations, return the best one
            nrNets = 1
            nnets = [NeuralNet(numLayers=6, nrNeurons=20) for _ in range(nrNets)]
            optimizers = [optim.Adam(nnet.parameters(), lr=0.002) for nnet in nnets]  # Adam 
            #schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=0.0002) for optimizer in optimizers]
            # set a normal annealing without cosine
            schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7) for optimizer in optimizers]
            # nnets, bestNet, bestIndex = fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='dihedral', addRange=0.1, optimizers=optimizers, schedulers=schedulers, epochs=10000, periodicRange=[-180, 180])
            nnets, bestNet, bestIndex = fitGPNN(gp, nnets, x, y, name, plot_directory, plotType='dihedral', addRange=0.1, optimizers=optimizers, schedulers=schedulers, xlabel='dihedral angle [-pi, pi]', epochs=3000, periodicRange=[-np.pi, np.pi])


            # save the neural net parameters
            self.prior_dihedral[name] = {'nnets': nnets, 'bestNet': bestNet, 'bestIndex': bestIndex, 'offset': popt[0].tolist()}
            
        return self.prior_dihedral
    
    
