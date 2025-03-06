# xyzToPdb
Convert `.xyz` to `.pdb` by guessing the lacking residues. 


**Example:**
Convert 2xp2 from `.pdb` to `.xyz` and back to `.pdb` (should work with any pdb).  
``` 
# from terminal
pip install xyzToPdb
wget https://files.rcsb.org/view/2xp2.pdb
python -c "from ase.io import read,write; write('2xp2.xyz', read('2xp2.pdb'))"
xyzToPdb 2xp2.xyz # writes 2xp2.xyz.pdb
```

**Compute Error:**
```
from ase.io import read
import numpy as np 

a = read('2xp2.pdb').arrays['residuenames'][:2262]
b = read('2xp2.xyz.pdb').arrays['residuenames'][:2262]

print(f"Corect={np.sum(a==b)}/{a.size}={np.mean(a==b)*100:.4f}%")

print("wrong predictions:\t", b[a!=b])
print("ground truth: \t\t", a[a!=b])
print("residue index:\t\t", np.arange(2262)[a!=b])
```
![image](https://github.com/user-attachments/assets/8f95678c-1d35-4ff8-82f3-8bbddfbc8624)


**Limitations:** xyzToPdb uses a neural network trained on PDB with ~99% validation accuracy. The more your `input.xyz` file differs to the NNs training on PDB, the worse it'll work. For example, if your `file.xyz`
 - is from a protein family not represented in PDB 
 - has had missing residues/atoms added (never happened in training)

Inspired by [this stackexchange post](https://mattermodeling.stackexchange.com/questions/9844/is-it-possible-to-recover-the-protein-structure-after-conversions-pdb-xyz-pdb).

## Training
Used 80% of PDB anno 2024. Model got ~99% `residue_acc` and ~99% `ca_cb_acc` on the 20% validation set. 

![image](https://github.com/user-attachments/assets/dbc2a6f9-09e0-4faa-a331-c6cb7d5e9c91)
