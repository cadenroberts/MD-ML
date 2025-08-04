#!/usr/bin/env python3
import numpy as np
import argparse

# Mostly written by ChatGPT

def compare_numpy_files(file1, file2, verbose):
    array1 = np.load(file1)
    array2 = np.load(file2)

    if array1.shape != array2.shape:
        print(f"Array shapes differ: {array1.shape} vs {array2.shape}")
        return

    if np.array_equal(array1, array2):
        if verbose:
            print("Arrays are identical")
    else:
        diff = array1 - array2
        num_differences = np.count_nonzero(diff)
        rms_diff = np.sqrt(np.mean(diff**2))
        max_diff = np.max(np.abs(diff))
        print(f"{num_differences}/{diff.size} elements differ, RMSD={rms_diff}, max(diff)={max_diff}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the contents fo two npy files.")
    parser.add_argument("file1", type=str, help="First numpy file")
    parser.add_argument("file2", type=str, help="Second numpy file")
    parser.add_argument("-s", action="store_true", help="Report when two files are the same")
    args = parser.parse_args()

    compare_numpy_files(args.file1, args.file2, args.s)