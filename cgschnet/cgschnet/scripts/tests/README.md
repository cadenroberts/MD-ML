## test_priors.sh and diff_priors.sh
These scripts expect to be run from the `cgschnet/scripts` directory. \
`./tests/test_priors.sh <INPUT_DATA_DIR> <OUTPUT_PATH>` \
This will generate several datasets in OUTPUT_PATH each using a different prior. \
`./tests/diff_priors.sh <OUTPUT_PATH_A> <OUTPUT_PATH_B> <PDBID>` \
This will compare all the prior output files for two of `test_priors.sh` for a given pdbid within the dataset.