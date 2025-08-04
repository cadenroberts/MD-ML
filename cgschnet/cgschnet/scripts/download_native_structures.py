#!/usr/bin/env python3
from Bio import PDB
import os
import logging
from gen_benchmark import machines
import mdtraj
from pathlib import Path

output_dir: Path = machines["bizon"].rmsd_dir
protein_ids: dict[str, str] = {
    #copied from paper https://pubmed.ncbi.nlm.nih.gov/37714883/
    "chignolin": "1UAO",
    "trpcage": "2JOF",
    "bba": "1FME",
    "villin": "2F4K",
    "wwdomain": "1PIN",
    "ntl9": "2HBA",
    "bbl": "2WXC",
    "proteinb": "1PRB",
    "homeodomain": "1ENH",
    "proteing": "1MI0",
    "a3d": "2A3D",
    "lambda": "1LMB"
}

special_slice: dict[str, list[str]] = {
    # get these strings like this:
    #thingy = mdtraj.load("/media/DATA_18_TB_1/andy/benchmark_set_5/wwdomain/starting_pos_0/processed/starting_pos_0_processed.pdb")
    #[thingy.topology.residue(i).code for i in range(thingy.topology.n_residues)]
    "wwdomain": list("KLPPGWEKRMSRSSGRVYYFNHITNASQWERPSG"),
    "proteinb": list("LKNAIEDAIAELKKAGITSDFYFNAINKAKTVEEVNALVNEILKAHA"),
    "proteing": list("DTYKLVIVLNGTTFTYTTEAVDAATAEKVFKQYANDAGVDGEWTYDAATKTFTVTE"),
    "lambda": list("PLTQEQLEDARRLKAIYEKKKNELGLSQESVADKMGMGQSGVGALFNGINALNAYNAALLAKILKVSVEEFSPSIAREIY")
}


def find_num_mismatches(subarray: list, array: list, start: int) -> int:
    count = 0
    for i in range(0, len(subarray)):
        if subarray[i] != array[start + i]:
            count += 1
    return count

def find_closest_subarray_indicies(subarray: list, array: list) -> tuple[int, list[int]]:
    closeness_per_index = [find_num_mismatches(subarray, array, start_index)
                         for start_index in range(0, len(array) - len(subarray) + 1)]

    closest = min(closeness_per_index)
    output = [i for i, v in enumerate(closeness_per_index) if v == closest]
        
    return closest, output
                
pdbl = PDB.PDBList() #pyright: ignore[reportPrivateImportUsage]

for protein_name, pdb_id in protein_ids.items():
    filename = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=output_dir)
    
    output_path = output_dir.joinpath(f"{protein_name}.pdb")
    
    if protein_name in special_slice.keys():
        temp_out_path = os.path.join(output_dir, f"{protein_name}_unsliced.pdb")
        os.rename(filename, temp_out_path)
        to_slice = mdtraj.load(temp_out_path)

        slice_to_match = special_slice[protein_name]
        
        dist, start_indicies = find_closest_subarray_indicies(
            slice_to_match,
            [to_slice.topology.residue(i).code for i in range(to_slice.topology.n_residues)]
        )

        if len(start_indicies) != 1:
            logging.warning(f"multiple positions with the same distance score of {dist}, choosing first one in list of starting positions {start_indicies}")
        
        start_index = start_indicies[0]

        sliced = to_slice.atom_slice(to_slice.topology.select(f"({start_index} <= resid) and (resid < {start_index + len(slice_to_match)})"))

        if dist != 0:
            logging.warning(f"no perfect slice match for protein {protein_name}, closest match has {dist} wrong amino acids out of a subslice length of {len(slice_to_match)}")
            sliced_str = [sliced.topology.residue(i).code for i in range(sliced.topology.n_residues)]
            logging.warning(f"slice        = {"".join(slice_to_match)}")
            logging.warning(f"experimental = {"".join(sliced_str)}")
            logging.warning(f"difference   = {"".join(["+" if sliced_str[i] != slice_to_match[i] else "-" for i in range(len(sliced_str))])}")

        sliced.save_pdb(output_path)
    else:
        os.rename(filename, output_path)
    
    logging.info(f"saved {pdb_id} to {output_path}")

