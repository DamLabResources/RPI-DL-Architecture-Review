import numpy as np
import torch
import torch.nn

class OneHotEncoder(nn.Module):
    """
    Layer to One-hot encode sequences
    """
    
    def __init__(self, molecule):
        
        assert molecule in ['rna','protein']
        
        self.type = molecule
        
        def _create_protein_dict():
            aas = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
            zeros = np.eye(20)
            
            return {aa: zeros[n,:] for n,aa in enumerate(aas)}
        
        self.dict = {'rna':     {'A': [1, 0, 0, 0],
                                 'U': [0, 1, 0, 0],
                                 'T': [0, 1, 0, 0],
                                 'G': [0, 0, 1, 0],
                                 'C': [0, 0, 0, 1],
                                 'N': [0.25]*4},
                     
                     'protein': _create_protein_dict()}
                     
    def forward(x):
        return torch.tensor([self.dict[self.type][letter] for letter in x])