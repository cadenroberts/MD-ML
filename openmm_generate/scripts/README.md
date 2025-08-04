## batchgenerate.py

This script runs OpenMM simulations for a set of proteins. It operates on either a json array file of pdbids, or a directory of .pdb files as specified using the "--input-dir=" option.

**Examples of use:**

Run the first PDB in openmm_2024.01.03_N1000.json for 1000 frames, removing ligands: \
`./batch_generate.py openmm_2024.01.03_N1000.json --prepare --remove-ligands --data-dir=../data --steps 1000`

Run all PDBs in openmm_2024.01.03_N1000.json for 1000 frames, with ligands: \
`./batch_generate.py openmm_2024.01.03_N1000.json --prepare --data-dir=../data --steps 1000`

Run all PDBs in openmm_2024.01.03_N1000.json for 1000 frames, with ligands, at 350k and with an 4fs timestep: \
`./batch_generate.py openmm_2024.01.03_N1000.json --prepare --data-dir=../data --steps 1000 --integrator=integrator_350k_4fs.json`

Run all pdb files in "../data/input/" for 10000 frames, saving every 10th frame: \
`./batch_generate.py --prepare --steps 10000 --report-steps=10 --input-dir=../data/input/ --data-dir=../data/output`

**Important options:**

`--gpus` : Distribute jobs over multiple gpus as specified by nvidia device IDs, e.g. --gpus=0,1,2,3 \
`--prepare` : Prepare pdb files (add solvent, clean up broken residues), you should always specify this option if you haven't prepared the files using another script. \
`--steps` : How many total frames to run. \
`--report-steps` : How often to save frames. \
`--help` : For a full list of options.