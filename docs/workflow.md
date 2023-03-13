# Workflow

This section gives a broad overview of the steps taken in processing images.

## Topotracing : Processing a single `.spm` file

Topotracing loads images from `.spm` files and extracts the specified channel, performing various filtering stages
(`Filters()` class) before finding grains (`Grains()` class) and then calculating statistics for each grain
(`GrainStats()` class). The Gaussian filtered image and labelling of grains is then passed onto DNA Tracing.

```{mermaid}
%%{init: {'theme': 'base',
         }
}%%
graph TD;

  subgraph Background Flattening
  A1([Load YAML Configuration]) --> A2([Load SPM])
  A2 --> A3([Extract channel from SPM])
  A3 --> A4([Initial Align])
  A4 --> A5([Initial Tilt Removal])
  A5 --> A6([Thresholding Otsu])
  A6 --> A7([Mask Generation])
  A7 --> A8([Masked Align])
  A8 --> A9([Masked Tilt Removal])
  A9 --> A10([Background Zeroing])
  end
  subgraph Grain Finding
  A10 --> B1([Lower Thresholding])
  B1 --> B2([Guassian Filtering])
  B2 --> B3([Tidy Edges])
  B3 --> B4([Preliminary Statistics])
  B4 --> B5([Size Thresholding])
  B5 --> B6([Label Regions])
  end
  subgraph Grain Statistics
  B6 --> C2([Calculate Points])
  C2 --> C8([Height & Volume Statistics])
  C2 --> C3([Calculate Edges])
  C2 --> C4([Calculate Centroid])
  C3 --> C5([Calculate Radius Statistics])
  C3 --> C6([Convex Hull / Graham Scan])
  C6 --> C7([Minimum Bounding Box Statistics])
  end
  subgraph DNA Tracing
  B2 --> D1([More Analysis])
  B5 --> D1
  end
  style A1 fill:#648FFF,stroke:#000000
  style A2 fill:#648FFF,stroke:#000000
  style A3 fill:#648FFF,stroke:#000000
  style A4 fill:#648FFF,stroke:#000000
  style A5 fill:#648FFF,stroke:#000000
  style A6 fill:#648FFF,stroke:#000000
  style A7 fill:#648FFF,stroke:#000000
  style A8 fill:#648FFF,stroke:#000000
  style A9 fill:#648FFF,stroke:#000000
  style A10 fill:#648FFF,stroke:#000000
  style B1 fill:#DC267F,stroke:#000000
  style B2 fill:#DC267F,stroke:#000000
  style B3 fill:#DC267F,stroke:#000000
  style B4 fill:#DC267F,stroke:#000000
  style B5 fill:#DC267F,stroke:#000000
  style B6 fill:#DC267F,stroke:#000000
  style C2 fill:#FE6100,stroke:#000000
  style C3 fill:#FE6100,stroke:#000000
  style C4 fill:#FE6100,stroke:#000000
  style C5 fill:#FE6100,stroke:#000000
  style C6 fill:#FE6100,stroke:#000000
  style C7 fill:#FE6100,stroke:#000000
  style C8 fill:#FE6100,stroke:#000000
  style D1 fill:#785EF0,stroke:#000000
```

## DNA Tracing : Processing a single grain

```{mermaid}
%%{init: {'theme': 'base',
         }
}%%
graph TD;

  subgraph dnaTrace
  A1(["get_numpy_arrays() | Load Numpy arrays to dictionary indexed by number"]) --> A2(["skimage.filters.gaussian() | Filter full image"])
  A2 -- For each image --> A3(["get_disordered_trace() | extracts mask"])
  A3 --> A4(["purge_obvious_crap() | Removes ites if len() < 10 (i.e. small objects) "])
  A4 --> A5(["linear_or_circular on unordered traces() | linear or circular molecule"])
  A5 --> A6(["get_ordered_traces() | Reorders points in the array?"])
  A6 --> A7(["linear_or_circular() on ordered traces"])
  A7 --> A8(["get_fitted_traces()"])
  A8 --> A9(["get_splined_traces()"])
  A9 --> A10(["measure_contour_length()"])
  A10 --> A11(["measure_end_to_end_distance()"])
  A11 --> A12(["report_basic_stats()"])
  end

  subgraph "get_disordered_trace()"
  A3 --> B1(["ndimage.binary_dilation() | extracts mask"])
  B1 --> B2(["scipy.ndimage.gaussian_filter molecule()"])
  B2 --> B3(["getSkeleton()"])
  B3 --> A4
  end

  subgraph "getSkeleton()"
  B3 --> C1(["Skeletonize | to be replaced by get_skeleton()"])
  C1 --> B3
  end

  style A1 fill:#648FFF,stroke:#000000
  style A2 fill:#648FFF,stroke:#000000
  style A3 fill:#648FFF,stroke:#000000
  style A4 fill:#648FFF,stroke:#000000
  style A5 fill:#648FFF,stroke:#000000
  style A6 fill:#648FFF,stroke:#000000
  style A7 fill:#648FFF,stroke:#000000
  style A8 fill:#648FFF,stroke:#000000
  style A9 fill:#648FFF,stroke:#000000
  style A10 fill:#648FFF,stroke:#000000
  style A11 fill:#648FFF,stroke:#000000
  style A12 fill:#648FFF,stroke:#000000
  style B1 fill:#DC267F,stroke:#000000
  style B2 fill:#DC267F,stroke:#000000
  style B3 fill:#DC267F,stroke:#000000
  style C1 fill:#FE6100,stroke:#000000

```
