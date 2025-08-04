#!/bin/bash
set -Euo pipefail
set -x
echo $1 $2 $3

NP_DIFF="$(dirname $0)/numpy_diff.py"

# for PRIOR in "Prior_CA" "Prior_CACB" "Prior_CA_lj" "Prior_CA_lj_angle"
for PRIOR in "Prior_CA" "Prior_CACB" "Prior_CA_lj_angle_dihedral" "Prior_CA_lj_angleXCX_dihedralX"; do
  # Compare the two prior files for protein $3 in preprocess outputs $1 and $2
  diff $1/$PRIOR/priors.yaml  $2/$PRIOR/priors.yaml
  diff $1/$PRIOR/prior_params.json  $2/$PRIOR/prior_params.json
  diff $1/$PRIOR/$3/processed/$3_processed.psf  $2/$PRIOR/$3/processed/$3_processed.psf
  $NP_DIFF $1/$PRIOR/$3/raw/embeddings.npy  $2/$PRIOR/$3/raw/embeddings.npy
  $NP_DIFF $1/$PRIOR/$3/raw/deltaforces.npy  $2/$PRIOR/$3/raw/deltaforces.npy
  $NP_DIFF $1/$PRIOR/$3/raw/coordinates.npy  $2/$PRIOR/$3/raw/coordinates.npy
  $NP_DIFF $1/$PRIOR/$3/raw/forces.npy  $2/$PRIOR/$3/raw/forces.npy
  #FIXME: Should check box.npy too if it exists
done
