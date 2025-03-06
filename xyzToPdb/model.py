import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

input_dim = 3  
hidden_dim = 256
num_layers = 6
num_heads = 8
num_amino_acids = 21 # 20 standard amino acids + 1 for non-amino acid

# Define amino acid mapping
def get_amino_acid_mapping():
    standard_aa = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
        'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]
    return {aa: i for i, aa in enumerate(standard_aa)}

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=10000):
        super().__init__()
        pe = th.zeros(max_len, hidden_dim)
        position = th.arange(0, max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        return x + self.pe[:x.size(1)].unsqueeze(0)


class SimpleAtomTransformer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=6, num_heads=8, num_amino_acids=21, 
                 dropout=0.1, num_atom_types=7, num_atom_names=36):
        super().__init__()
        
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.position_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Use standard transformer encoder layers
        self.layers = th.nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        ) for _ in range(num_layers)])
        
        self.output_projection = nn.Linear(hidden_dim, num_amino_acids)
        self.atom_name_projection = nn.Linear(hidden_dim, num_atom_names)
        # Add new head for residue change prediction
        self.residue_change_projection = nn.Linear(hidden_dim, 1)
        
    def forward(self, atom_types, atom_positions, mask=None):
        # Extract atom types and positions
        
        atom_embeddings = self.atom_embedding(atom_types)
        position_embeddings = self.position_projection(atom_positions)
        h = atom_embeddings + position_embeddings
        h = self.positional_encoding(h)
        
        # Pass positions to each layer for distance calculation
        for layer in self.layers:
            #with th.amp.autocast(device_type='cuda', dtype=th.bfloat16):
            h = layer(h, mask, is_causal=False)
        
        amino_acid_logits = self.output_projection(h)
        atom_name_logits = self.atom_name_projection(h)
        # Get residue change logits
        residue_change_logits = self.residue_change_projection(h).squeeze(-1)
        
        amino_acid_probs = F.softmax(amino_acid_logits, dim=-1)
        atom_name_probs = F.softmax(atom_name_logits, dim=-1)
        # Apply sigmoid for binary classification
        residue_change_probs = th.sigmoid(residue_change_logits)
        
        return amino_acid_probs, atom_name_probs, residue_change_probs

# Define a mapping from atom symbols to indices
def get_atom_type_mapping():
    common_atoms = ['C', 'N', 'O', 'S', 'H', 'P']
    return {atom: i for i, atom in enumerate(common_atoms)}

# Add this function to define atom name mapping
def get_atom_name_mapping():
    # Common atom names in PDB files
    common_atom_names = ['CA', 'CB', 'C', 'N', 'O', 'CG', 'CD', 'NE', 'CZ', 'NH1', 
                         'NH2', 'CD1', 'CD2', 'CE', 'OG', 'OG1', 'CG1', 'CG2', 'OD1', 
                         'OD2', 'ND1', 'ND2', 'NE2', 'CE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 
                         'CH2', 'OH', 'SD', 'SG', 'OE1', 'OE2', 'NZ']
    return {name: i for i, name in enumerate(common_atom_names)}

