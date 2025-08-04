import os
import requests

with open("input.txt","rt") as f:
    pdb_list = [i for i in f.read().strip().split(",") if i]

print(len(pdb_list))

os.makedirs("pdbs", exist_ok=True)

for pdbid in pdb_list:
    pdb_path = f"pdbs/{pdbid}.pdb.gz"
    pdb_url = f"https://files.rcsb.org/download/{pdbid}.pdb.gz"

    try:
        # download pdb file
        if not os.path.exists(pdb_path):
            r = requests.get(pdb_url)
            r.raise_for_status()
            with open(pdb_path, "wb") as f:
                f.write(r.content)
            print(f"{pdbid}")
        else:
            print(f"{pdbid}.pdb already downloaded")
    except Exception as e:
        print(e)