# xyzToPdb
Convert `.xyz` to `.pdb` (guess residues). 

**Problem:**
  Someone converted `file.pdb` to `file.xyz` but only gave you `file.xyz` => you can't visualize the protein(s) in `file.xyz` (because it has no residue information).

**Solution:** `pip install xyzToPdb; xyzToPdb file.xyz # outputs file.pdb`

Inspired by [this stackexchange post](https://mattermodeling.stackexchange.com/questions/9844/is-it-possible-to-recover-the-protein-structure-after-conversions-pdb-xyz-pdb).

## How? 
Runs a Transformer trained with the loss `cross_entropy(Transformer(file.xyz) - file.pdb)` on 80% of PDB anno 2024. The model got ~99% `residue_acc` and ~99% `ca_cb_acc` on the 20% validation set. 

![image](https://github.com/user-attachments/assets/dbc2a6f9-09e0-4faa-a331-c6cb7d5e9c91)
