import mdtraj
import glob
import tqdm
import os

with open("sequences.txt","wt",encoding="utf-8") as seq_file:
  for i in tqdm.tqdm(sorted(glob.glob("pdbs/*.pdb.gz"))):
    traj = mdtraj.load(i)
    l = []
    for c in traj.top.chains:
      cseq = "".join([r.code if r.code else "!" for r in c.residues if not r.is_water])
      if cseq:
        l.append(cseq)
      pdbid = os.path.basename(i)[:-len(".pdb.gz")]
    print(pdbid+":",",".join(l), file=seq_file)
    
    
    
