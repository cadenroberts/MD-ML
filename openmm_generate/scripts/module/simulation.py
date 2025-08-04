from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np
import h5py
import os
import math
import json
from sys import stdout
from module import ligands
from module import function
from module.reporters import ExtendedH5MDReporter

def run(pdbid=str, input_pdb_path=str, steps=100, report_steps=1, load_ligand_smiles=True, atomSubset=None,
        resume_checkpoint=False, interrupt_callback=None, integrator_params=None):
    """
    Run the simulation for the given PDB ID.

    Args:
        pdbid (str): The PDB ID.
        input_pdb_path (str): The path to the input PDB file.
        steps (int): The total number of steps to run.
        report_steps (int): Data is written to the h5 file every 'report_steps' steps.
        load_ligand_smiles (bool): If true load the ligand templates saved during prepare.
        atomSubset (list or None): List of atom indices to subset. Defaults to None.
        resume_checkpoint (bool): If true try to resume from an existing checkpoint file, if false
            start a new simulatin.
        interrupt_callback (function): This function is called at every checkpoint interval (<1000 steps),
            if it returns false then the simmulation is gracefully stopped.
        integrator_params (dict): A dict of integrator parameters to modify, possibile keys:
            dt, temperature, friction, pressure, barostatInterval

    Returns:
        (int): The last step completed.
    """
    
    
    print(f"Start simulation of {pdbid}...")
    
    # Input Files
    pdb = PDBFile(input_pdb_path)

    ff_configs = ['amber14-all.xml', 'amber14/tip3pfb.xml']
    ff_configs_path = function.get_data_path(f'{pdbid}/processed/forcefield.json')
    if os.path.exists(ff_configs_path):
        ff_configs = json.load(open(ff_configs_path, 'r', encoding='utf-8'))
    print("Forcefield:", ff_configs)
    implicit_solvent = any(["implicit" in i for i in ff_configs])

    forcefield = ForceField(*ff_configs)

    if load_ligand_smiles:
        input_ligands_path = os.path.splitext(input_pdb_path)[0]+"_ligands_smiles.json"
        template_cache_path = os.path.splitext(input_pdb_path)[0]+"_ligands_cache.json"
        if os.path.exists(input_ligands_path):
            ligands.add_ff_template_generator_from_json(forcefield, input_ligands_path, template_cache_path)
        else:
            print(f"'{input_ligands_path}' does not exist, skipping template generation.")

    # System Configuration
    nonbondedMethod = PME
    nonbondedCutoff = 1.0*nanometers
    ewaldErrorTolerance = 0.0005
    constraints = HBonds
    rigidWater = True
    constraintTolerance = 0.000001
    hydrogenMass = 1.5*amu

    # Integration Options
    if integrator_params is None:
        integrator_params = dict()
    dt = integrator_params.get("dt", 0.002*picoseconds)
    temperature = integrator_params.get("temperature", 300*kelvin)
    friction = integrator_params.get("friction", 1.0/picosecond)
    pressure = integrator_params.get("pressure", 1.0*atmospheres)
    barostatInterval = integrator_params.get("barostatInterval", 25)

    # Additional values for implicit solvent:
    if implicit_solvent:
        saltCon = 0.15 # unit.molar
        solventDielectric = 78.5 # Default solvent dielectric: http://docs.openmm.org/latest/userguide/application/02_running_sims.html @ 2024.02.11
        implicitSolventKappa = 7.3*50.33355*math.sqrt(saltCon/solventDielectric/temperature.value_in_unit(kelvin))*(1/nanometer)

    # Simulation Options
    equilibrationSteps = int((10*picoseconds)/dt)
    checkpointInterval = min(int(steps/10), 1000)
    platformNames = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    if 'CUDA' in platformNames:
        platform = Platform.getPlatformByName('CUDA')
        platformProperties = {'Precision': 'single'}
    elif 'OpenCL'in platformNames:
        platform = Platform.getPlatformByName('OpenCL')
        platformProperties = {'Precision': 'single'}
    else:
        platform = None
        platformProperties = {}
    print(f"Simulation platform: {platform.getName()}, {platformProperties}")
    
    if resume_checkpoint:
        with h5py.File(function.get_data_path(f"{pdbid}/result/output_{pdbid}.h5"), "r") as f:
            last_recorded_time = f["time"][-1]

    # Reporters
    hdf5Reporter = None
    try:
        hdf5Reporter = ExtendedH5MDReporter(function.get_data_path(f'{pdbid}/result/output_{pdbid}.h5'), report_steps, total_steps=steps,
                                            atom_subset=atomSubset, append_file=resume_checkpoint)
        dataReporter = StateDataReporter(function.get_data_path(f'{pdbid}/simulation/log.txt'), checkpointInterval, totalSteps=steps,
            step=True, speed=True, progress=True, potentialEnergy=True, temperature=True, separator='\t')
        checkpointReporter = CheckpointReporter(function.get_data_path(f'{pdbid}/simulation/checkpoint.chk'), checkpointInterval)

        # Prepare the Simulation
        topology = pdb.topology
        positions = pdb.positions

        if not resume_checkpoint:
            print('Building system...')
            if implicit_solvent:
                system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, constraints=constraints, hydrogenMass=hydrogenMass,
                                                 implicitSolventKappa=implicitSolventKappa)
            else:
                system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
                    constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance, hydrogenMass=hydrogenMass)
                system.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))
            integrator = LangevinMiddleIntegrator(temperature, friction, dt)
            integrator.setConstraintTolerance(constraintTolerance)
            simulation = Simulation(topology, system, integrator, platform, platformProperties)
            simulation.context.setPositions(positions)

            # Write XML serialized objects
            with open(function.get_data_path(f"{pdbid}/simulation/system.xml"), mode="w") as file:
                file.write(XmlSerializer.serialize(system))
            with open(function.get_data_path(f"{pdbid}/simulation/integrator.xml"), mode="w") as file:
                file.write(XmlSerializer.serialize(integrator))

            # Minimize and Equilibrate
            print('Performing energy minimization...')
            simulation.minimizeEnergy()
            print('Equilibrating...')
            simulation.context.setVelocitiesToTemperature(temperature)
            simulation.step(equilibrationSteps)
            simulation.currentStep = 0

        else: # resume_checkpoint
            print('Loading saved system...')
            system = function.get_data_path(f"{pdbid}/simulation/system.xml")
            integrator = function.get_data_path(f"{pdbid}/simulation/integrator.xml")
            simulation = Simulation(topology, system, integrator, platform, platformProperties)
            simulation.loadCheckpoint(function.get_data_path(f'{pdbid}/simulation/checkpoint.chk'))

            # Validate that the checkpoint steps matches the last step recorded in the h5 file
            last_simulation_time = simulation.context.getTime().value_in_unit(picoseconds)
            if not math.isclose(last_simulation_time, last_recorded_time, rel_tol=1e-3):
                raise RuntimeError(f"Simulation time does not match last recorded time: {last_simulation_time} != {last_recorded_time}")

            print(f"Resuming simulation from step {simulation.currentStep}...")

        # Simulate
        print('Simulating...')
        simulation.reporters.append(hdf5Reporter)
        simulation.reporters.append(dataReporter)
        simulation.reporters.append(checkpointReporter)

        simulation.reporters.append(StateDataReporter(stdout, checkpointInterval, step=True,
            progress=True, remainingTime=True, speed=True, totalSteps=steps, separator="\t"))

        # Propagate the simulation and save the data
        if not interrupt_callback:
            simulation.step(steps - simulation.currentStep)
        else:
            while (steps - simulation.currentStep) > 0:
                if not interrupt_callback():
                    print(f"Simulation of {pdbid} interrupted at step {simulation.currentStep}")
                    return simulation.currentStep
                simulation.step(min(steps - simulation.currentStep, checkpointInterval))
    finally:
        # close the reporters
        if hdf5Reporter:
            hdf5Reporter.close()

    # Write file with final simulation state
    simulation.saveState(function.get_data_path(f"{pdbid}/simulation/final_state.xml"))
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=simulation.system.usesPeriodicBoundaryConditions())
    with open(function.get_data_path(f"{pdbid}/simulation/final_state.pdb"), mode="w") as file:
        PDBFile.writeFile(simulation.topology, state.getPositions(), file)

    # Assert the data
    with h5py.File(function.get_data_path(f"{pdbid}/result/output_{pdbid}.h5"), "r") as f:
        # Check the shape of the data
        assert f["coordinates"].shape == f["forces"].shape
        # Check the dimension of the data
        for key in ["coordinates", "forces"]:
            print(key, f[key].shape)
            assert f[key].shape[0] == steps//report_steps
            assert f[key].shape[1] == len(atomSubset)
            assert f[key].shape[2] == 3

    print(f"Simulation of {pdbid} is done.")
    print(f"Result is here: {function.get_data_path(f'{pdbid}/result/output_{pdbid}.h5')}\n")

    return simulation.currentStep
