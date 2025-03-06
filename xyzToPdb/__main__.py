import sys
import os
import torch
from ase.io import read, write
from ase import Atoms
import numpy as np
import urllib.request
import time
from xyzToPdb.model import (
    SimpleAtomTransformer, 
    get_atom_type_mapping, 
    get_atom_name_mapping, 
    get_amino_acid_mapping,
    input_dim, 
    hidden_dim, 
    num_layers, 
    num_heads
)

def download_model(model_path):
    print(f"Model not found at {model_path}, downloading...")
    try:
        model_url = "https://logmd.b-cdn.net/model.pth"
        
        # Remove existing model file if it exists
        if os.path.exists(model_path):
            os.remove(model_path)
            
        # Create a custom reporthook to show download progress
        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write(f"\r...{percent}% - {progress_size / (1024 * 1024):.1f} MB "
                            f"of {total_size / (1024 * 1024):.1f} MB "
                            f"({speed} KB/s)")
            sys.stdout.flush()
            
        urllib.request.urlretrieve(model_url, model_path, reporthook)
        print(f"\nModel downloaded successfully to {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

def main():
    print("Assuming amino acids appear sequentially (including their hydrogens if they exist).")
    print("Assuming water molecules are in the end. ")
    # Check if input file is provided
    if len(sys.argv) < 2:
        print("Usage: xyztopdb file.xyz")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Determine output filename
    output_file = input_file + ".pdb"
    output_file_only_protein = input_file + "_only_protein.pdb"
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    if not os.path.exists(model_path):
        download_model(model_path)
    else: 
        import hashlib; 
        model_hash = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
        real_hash = '6b92245812e171207f9abbca11da429ec448e2a36a158450e39ef9ab413904f0'
        if model_hash != real_hash:
            print(f"Model hash: {model_hash}")
            print(f"Should be:  {real_hash}")
            print("Likely just new model, downloading...")
            download_model(model_path)
        
        
    model = SimpleAtomTransformer(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        num_heads=num_heads, 
        num_amino_acids=21, 
        num_atom_types=len(get_atom_type_mapping()) + 1
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Load and process XYZ file - 
    # `atoms` will be prepared for inference by removing hydrogens/waters. 
    # we then re-annotate raw_atoms using model(atoms) residue information. 
    atoms = read(input_file)
    raw_atoms = read(input_file)

    # Remove water molecules (vectorized approach)
    symbols = np.array(atoms.symbols)
    o_indices = np.where(symbols == 'O')[0]
    h_indices = np.where(symbols == 'H')[0]
    distances = np.linalg.norm(atoms.positions[h_indices, None, :] - atoms.positions[None, o_indices, :], axis=2)
    water_o_indices = o_indices[np.sum(distances < 1.2, axis=0) == 2]
    water_h_indices = h_indices[np.any(distances[:, np.sum(distances < 1.2, axis=0) == 2] < 1.2, axis=1)]

    non_water_and_h_indices = np.array( [i for i in range(len(atoms)) if i not in np.concatenate([water_o_indices, water_h_indices, h_indices])])
    atoms = atoms[non_water_and_h_indices]
    
    # Get atom positions and symbols
    positions = atoms.positions.astype(np.float32)
    symbols = atoms.symbols
    
    # Convert symbols to indices
    atom_types = []
    for symbol in [a for a in symbols]:
        atom_types.append(get_atom_type_mapping().get(symbol, len(get_atom_type_mapping())))
    
    atom_types = torch.tensor(atom_types).reshape(1, -1).long()
    atom_positions = torch.tensor(positions).reshape(1, -1, 3)
    
    # Run inference
    with torch.no_grad():
        print(atom_types.shape, atom_positions.shape)
        amino_acid_probs, atom_name_probs, atom_change_probs = model(atom_types, atom_positions)
        _, predicted = torch.max(amino_acid_probs[0], dim=1)
        
        num_to_aas = list(get_amino_acid_mapping().keys()) + ['DUM']  # index 21 is not-amino-acid
        atom_residue_names = np.array([num_to_aas[i] for i in predicted])
        
        _, predicted_atom_names = torch.max(atom_name_probs[0], dim=1)
        num_to_atom_names = list(get_atom_name_mapping().keys()) + ['X']
        atom_names = np.array([num_to_atom_names[i] for i in predicted_atom_names])

        # changes = np.concatenate(([1], np.diff(residue_numbers) != 0)).astype(int)
        predicted_atom_changes = (atom_change_probs[0] > 0.5).numpy()
        atom_residue_ids = np.cumsum(predicted_atom_changes)

    print(atom_names, atom_residue_names, atom_residue_ids)
    atoms.arrays['atomtypes'] = atom_names
    atoms.arrays['residuenames'] = atom_residue_names
    atoms.arrays['residuenumbers'] = atom_residue_ids

    # write protein with residue information 
    write(output_file_only_protein, atoms, format='proteindatabank')
    
    # Add back hydrogen and water. 
    # We do this by taking raw_atoms with hydrogen/water and re-annotating with `atoms` residue info.
    raw_atoms.arrays['atomtypes'] = list(raw_atoms.symbols)
    raw_atoms.arrays['residuenames'] = ['DUM'] * len(raw_atoms)
    raw_atoms.arrays['residuenumbers'] = [-1] * len(raw_atoms)

    for i, idx in enumerate(non_water_and_h_indices):
        raw_atoms.arrays['atomtypes'][idx] = atom_names[i]
        raw_atoms.arrays['residuenames'][idx] = atom_residue_names[i]
        raw_atoms.arrays['residuenumbers'][idx] = atom_residue_ids[i]

    # Add hydrogens to closest by residue. 
    h_atoms = raw_atoms.positions[h_indices, None, :] 
    distances = atoms.positions[None, :, :] - h_atoms
    nearest_atom_indices = np.argmin(np.linalg.norm(distances, axis=2), axis=1)
    for i, h_idx in enumerate(h_indices):
        if np.min(distances[i]) <= 1.2:
            raw_atoms.arrays['residuenames'][h_idx] = atom_residue_names[nearest_atom_indices[i]]
            raw_atoms.arrays['residuenumbers'][h_idx] = atom_residue_ids[nearest_atom_indices[i]]

    # Fix hydrogen residue numbers to maintain increasing order
    for h_idx in h_indices:
        current_residue_num = raw_atoms.arrays['residuenumbers'][h_idx] 
        neighbour_before = [a for a in range(h_idx-3, h_idx) if a >= 1]
        neighbour_after  = [a for a in range(h_idx+1, h_idx+4) if a < len(raw_atoms)]

        neighbour_before_nums = np.array([raw_atoms.arrays['residuenumbers'][a] for a in neighbour_before ])
        neighbour_after_nums = np.array([raw_atoms.arrays['residuenumbers'][a] for a in neighbour_after ])

        if neighbour_after_nums.size == 0 or neighbour_before_nums.size == 0: continue 

        if neighbour_before_nums.min() == neighbour_after_nums.min(): 
            raw_atoms.arrays['residuenumbers'][h_idx] = neighbour_before_nums.min()
            raw_atoms.arrays['residuenames'][h_idx] = raw_atoms.arrays['residuenames'][neighbour_after[np.argmin(neighbour_after_nums)]]


    # Fix water count. 
    water_count = 1 + max(raw_atoms.arrays['residuenumbers'])
    o_indices = np.where(np.array(raw_atoms.symbols) == 'O')[0]
    for o_idx in o_indices:
        if raw_atoms.arrays['residuenames'][o_idx] == 'DUM':
            h_distances = np.linalg.norm(raw_atoms.positions[h_indices] - raw_atoms.positions[o_idx], axis=1)
            closest_h = h_indices[np.argsort(h_distances)[:2]]
            if len(closest_h) == 2 and np.all(h_distances[np.argsort(h_distances)[:2]] < 1.2):
                # This is likely a water molecule
                for h_idx in closest_h:
                    raw_atoms.arrays['residuenames'][h_idx] = 'HOH'
                    raw_atoms.arrays['residuenumbers'][h_idx] = water_count
                raw_atoms.arrays['residuenames'][o_idx] = 'HOH'
                raw_atoms.arrays['residuenumbers'][o_idx] = water_count
                water_count += 1

    write(output_file, raw_atoms, format='proteindatabank')
    print(f"wrote to {output_file}")

if __name__ == "__main__":
    main()