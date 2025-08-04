#!/bin/bash
set -Eeuo pipefail
set -x
echo $1 $2
# "${@:x}" is a bash trick that passes any remaining arguments (x onward) to the command
# https://stackoverflow.com/questions/3811345/how-to-pass-all-arguments-passed-to-my-bash-script-to-a-function-of-mine
./preprocess.py $1 -o $2/Prior_CA --prior CA --num_frames=10000 "${@:3}"
./preprocess.py $1 -o $2/Prior_CACB --prior CACB --num_frames=10000 "${@:3}"
#./preprocess.py $1 -o $2/Prior_CA_lj --prior CA_lj --num_frames=10000 "${@:3}"
# ./preprocess.py $1 -o $2/Prior_CA_lj_angle --prior CA_lj_angle --num_frames=10000 "${@:3}"
./preprocess.py $1 -o $2/Prior_CA_lj_angle_dihedral --prior CA_lj_angle_dihedral --num_frames=10000 "${@:3}"
./preprocess.py $1 -o $2/Prior_CA_lj_angleXCX_dihedralX --prior CA_lj_angleXCX_dihedralX --num_frames=10000 "${@:3}"
