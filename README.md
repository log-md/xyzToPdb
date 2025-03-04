# xyzToPdb
Convert `.xyz` to `.pdb` (guess residues). 

*Problem:*
  Someone `file.pdb` convert to `file.xyz` and only gives you `file.xyz` with no residue information => we can't plot it. 

*Solution:* `pip install xyzToPdb; xyzToPdb file.xyz # outputs file.pdb`

How? Runs a Transformer that was trained with the loss `cross_entropy(Transformer(file.xyz) - file.pdb)`. Wandb log here ~x% `residue_acc` and ~y% `ca_cb_acc`. 
