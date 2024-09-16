import numpy as np
import pandas as pd
from typing import Tuple

class dnaProteinComplex:
    def __init__(
            self,
            image: np.ndarray,
            grain: np.ndarray,
            filename: str,
            pixel_to_nm_scaling: float,
            convert_nm_to_m: bool = True,
        ):
        self.image = image * 1e-9 if convert_nm_to_m else image
        self.grain = grain
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling * 1e-9 if convert_nm_to_m else pixel_to_nm_scaling

    def store_multiclass_masks(self, dna_class = 1, protein_class = 2):
        mask = self.grain["above"]
        self.dna_binary_mask = mask[:, :, dna_class]
        self.protein_binary_mask = mask[:, :, protein_class]
        
        # Save masks to files
        np.save("/Users/laura/Desktop/mask_protein.npy", self.protein_binary_mask)
        np.save("/Users/laura/Desktop/mask_dna.npy", self.dna_binary_mask)

def run_dna_protein_analysis(
    image: np.ndarray,
    grain_masks: np.ndarray,
    filename: str,
    pixel_to_nm_scaling: float,
) -> Tuple[pd.DataFrame, None]:

    dna_protein = dnaProteinComplex(
        image=image,
        grain=grain_masks,
        filename=filename,
        pixel_to_nm_scaling=pixel_to_nm_scaling
    )
    dna_protein.store_multiclass_masks()

