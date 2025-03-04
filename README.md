# xyzToPdb
Convert `.xyz` to `.pdb` (guess residues). 

**Problem:**
  Someone converted `file.pdb` to `file.xyz` but only gave you `file.xyz` which has no residue information => can't visualize the protein(s) in `file.xyz`. 

**Solution:** `pip install xyzToPdb; xyzToPdb file.xyz # outputs file.pdb`

## How? 
Runs a Transformer trained with the loss `cross_entropy(Transformer(file.xyz) - file.pdb)`. Wandb log here ~x% `residue_acc` and ~y% `ca_cb_acc`. 
