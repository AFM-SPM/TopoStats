import numpy as np
import pandas as pd
from typing import Tuple
from skimage.measure import label, regionprops

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

    def isolate_connected_classes(self, dna_class=1, protein_class=2, combined=3):
        """ 
        Function that takes the full mask tensor and isolates the DNA and protein from the
        combined mask
        """
        mask = self.grain
        dna_mask = (mask[:, :, dna_class] > 0).astype(bool)
        protein_mask = (mask[:, :, protein_class] > 0).astype(bool)
        combined_mask = (mask[:, :, combined] > 0).astype(bool)
        
        # Label the combined mask
        labelled_combined_mask = label(combined_mask)
        
        result_dict = {}

        # Iterate over each labelled regions in the combined mask
        for region in regionprops(labelled_combined_mask):
            label_id = region.label
            
            # Extract bounding box for the region
            min_row, min_col, max_row, max_col = region.bbox
            
            # Extract the corresponding region from the DNA and protein masks
            combined_region = (labelled_combined_mask[min_row:max_row, min_col:max_col] == label_id)
            dna_region = (dna_mask[min_row:max_row, min_col:max_col])
            protein_region = (protein_mask[min_row:max_row, min_col:max_col])
            
            dna_only = np.logical_and(dna_region, combined_region) & ~protein_region
            protein_only = np.logical_and(protein_region, combined_region) & ~dna_region
            
            result_dict[label_id] = {
                'dna_only': dna_only,
                'protein_only': protein_only
            }

        return result_dict

    

