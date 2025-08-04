import os
import shutil
from tqdm import tqdm
import mdtraj as md

def delete_dirs_with_empty_or_invalid_h5(base_dir):
    """
    Checks for 'result' directories inside each subdirectory of base_dir.
    If a 'result' directory contains an invalid or no .h5 file, it deletes the parent directory.

    Args:
        base_dir (str): Path to the main directory containing subdirectories.
    """
    good, bad = 0, 0
    # Get list of subdirectories
    sub_dirs = os.listdir(base_dir)

    # Wrap the list of directories with tqdm for progress bar
    for sub_dir in tqdm(sub_dirs, desc="Checking directories", unit="dir"):
        full_sub_dir_path = os.path.join(base_dir, sub_dir)
        print(f'good: {good}, bad: {bad}')
        
        # Ensure it is a directory
        if os.path.isdir(full_sub_dir_path):
            result_dir = os.path.join(full_sub_dir_path, "result")
            
            # Check if 'result' directory exists
            if os.path.exists(result_dir) and os.path.isdir(result_dir):
                # Check for h5 file
                h5_files = [f for f in os.listdir(result_dir) if f.endswith(".h5")]
                if len(h5_files) == 1:
                    h5_file_path = os.path.join(result_dir, h5_files[0])
                    try:
                        # Attempt to open the h5 file with MDTraj
                        # print(f"Trying to open {h5_file_path} with MDTraj...")
                        trajectory = md.load(h5_file_path)
                        # print(f"Successfully opened {h5_file_path} with MDTraj.")
                        good += 1
                    except Exception as e:
                        # print(f"Failed to open {h5_file_path} with MDTraj: {e}")
                        # print(f"Deleting directory due to invalid h5 file: {full_sub_dir_path}")
                        shutil.rmtree(full_sub_dir_path)
                        bad += 1
                else:
                    # print(f"No valid h5 file found in {result_dir}, deleting directory.")
                    shutil.rmtree(full_sub_dir_path)
                    bad += 1

if __name__ == "__main__":
    # TODO: Add dir as arg
    base_directory = "/global/cfs/cdirs/m4229/jason/2_trajs"
    delete_dirs_with_empty_or_invalid_h5(base_directory)

