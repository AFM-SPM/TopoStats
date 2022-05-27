# Workflow

This section gives a broad overview of the steps taken in processing images.


## Topotracing : Processing a single `.spm` file.

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
  B2 --> D1([DNA Tracing])
  B3 --> B4([Preliminary Statistics])
  B4 --> B5([Size Thresholding])
  B5 --> C1([Label Regions])
  B5 --> D1
  end
  subgraph Grain Statistics
  C1 --> C2([Calculate Points])
  C2 --> C3([Calculate Edges])
  C2 --> C4([Calculate Centroid])
  C3 --> C5([Calculate Radius Statistics])
  C3 --> C6([Convex Hull / Graham Scan])
  end
  subgraph DNA Tracing
  D1 --> D2([More Analysis])
  end
  style A1 fill:#914800,stroke:#000000
  style A2 fill:#914800,stroke:#000000
  style A3 fill:#914800,stroke:#000000
  style A4 fill:#914800,stroke:#000000
  style A5 fill:#914800,stroke:#000000
  style A6 fill:#914800,stroke:#000000
  style A7 fill:#914800,stroke:#000000
  style A8 fill:#914800,stroke:#000000
  style A9 fill:#914800,stroke:#000000
  style A10 fill:#914800,stroke:#000000
  style B1 fill:#009110,stroke:#000000
  style B2 fill:#009110,stroke:#000000
  style B3 fill:#009110,stroke:#000000
  style B4 fill:#009110,stroke:#000000
  style B5 fill:#009110,stroke:#000000
  style C1 fill:#910007,stroke:#000000
  style C2 fill:#910007,stroke:#000000
  style C3 fill:#910007,stroke:#000000
  style C4 fill:#910007,stroke:#000000
  style C5 fill:#910007,stroke:#000000
  style C6 fill:#910007,stroke:#000000
  style D1 fill:#ba26f0,stroke:#000000
  style D2 fill:#ba26f0,stroke:#000000
```

## DNA Tracing : Processing a single grain
