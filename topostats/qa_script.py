import pandas as pd
from pathlib import Path
from topostats import TOPOSTATS_COMMIT
total_contour_length_minimum_threshold = 400e-9
total_contour_length_maximum_threshold = 460e-9

file_path = Path("/Volumes/shared/pyne_group/Shared/AFM_Data/dna_damage/Cs137_irradiations/20260204-analysis-getting-back-into-the-project/output_test_tracing_overhaul/grain_statistics.csv")

df = pd.read_csv(file_path)
num_successfully_processed = df["total_contour_length"].notnull().sum()
num_grains_total = len(df)
num_within_total_contour_length_threshold = df["total_contour_length"].between(
    total_contour_length_minimum_threshold, total_contour_length_maximum_threshold
).sum()
text = ""
text += f"TopoStats commit: {TOPOSTATS_COMMIT}\n"
text += f"Number of grains successfully processed: {num_successfully_processed} / {num_grains_total} ({num_successfully_processed / num_grains_total * 100:.2f}%)\n"
text += f"Number of successful grains with total_contour_length between {total_contour_length_minimum_threshold} and {total_contour_length_maximum_threshold}: {num_within_total_contour_length_threshold} / {num_successfully_processed} ({num_within_total_contour_length_threshold / num_successfully_processed * 100:.2f}%)\n"
print(text)
