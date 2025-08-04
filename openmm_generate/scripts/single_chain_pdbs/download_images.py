import os
import requests

save_dir = "pdb_images"

if not os.path.exists(save_dir):
  os.mkdir(save_dir)

with open("sequences2.txt","rt",encoding="utf-8") as seq_file:
  lines = seq_file.read().strip().split("\n")

for l in lines:
    pdbid = l.split(":")[0]
    pdbid = pdbid.lower()
    pdb_path = os.path.join(save_dir, f"{pdbid}_assembly-1.jpeg") 
    pdb_url = f"https://cdn.rcsb.org/images/structures/{pdbid}_assembly-1.jpeg"

    try:
        # download pdb file
        if not os.path.exists(pdb_path):
            r = requests.get(pdb_url)
            r.raise_for_status()
            with open(pdb_path, "wb") as f:
                f.write(r.content)
            print(f"{pdbid}")
        else:
            print(f"{pdbid} already downloaded")
    except Exception as e:
        print(e)