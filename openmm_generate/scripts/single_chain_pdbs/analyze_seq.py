
import Levenshtein

with open("sequences.txt","rt",encoding="utf-8") as seq_file:
  lines = seq_file.read().strip().split("\n")

print(len(lines),": original total")

interesting = []
seen = set()
for l in lines:
  # Remove anything that had heteromolecules
  if '!' in l:
    continue
  pdbid, seq = l.split(":")
  # Remove any exact duplicates
  if seq in seen:
    continue
  # Remove anything with less than 12 amino acids
  if len(seq) < 12:
    continue
  seen.add(seq)
  interesting.append([pdbid, seq])

print(len(interesting), ": remove duplicates, ligands, short")

# Remove all sequences that have greater than score_cutoff similarity
i = 0
while i < len(interesting):
  j = i + 1
  while j < len(interesting):
    # Levenshtein.ratio returns zero if the score is lower than score_cutoff
    if Levenshtein.ratio(interesting[i][1], interesting[j][1], score_cutoff=0.75) != 0:
      del[interesting[j]]
    else:
      j += 1
  i += 1

print(len(interesting), ": Levenshtein cutoff")

# interesting.sort(key=lambda x : x.split(":")[1])

with open("sequences2.txt","wt",encoding="utf-8") as seq_file:
  seq_file.write("\n".join([":".join(i) for i in interesting]))