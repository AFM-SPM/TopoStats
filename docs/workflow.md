# Workflow

This section gives a broad overview of the steps taken in processing images.


## Topotracing : Processing a single `.spm` file.

Topotracing loads images from `.spm` files and extracts the specified channel, performing various filtering stages
(`Filters()` class) before finding grains (`Grains()` class) and then calculating statistics for each grain
(`GrainStats()` class). The Gaussian filtered image and labelling of grains is then passed onto DNA Tracing.

```mermaid
x%%{INIT: { 'theme': 'base',
     'themeVariables':
         {'primaryColor': '#ffcccc',
          'secondaryColor': '#27e686',
          'tertiaryColor': 'e67c27'
         }
    }
}%%
graph TD;

  subgraph Background Flattening
  A1[Load YAML Configuration] --> A2[Load SPM]
  A2 --> A3[Extract channel from SPM]
  A3 --> A4[Initial Align]
  A4 --> A5[Initial Tilt Removal]
  A5 --> A6[Thresholding Otsu]
  A6 --> A7[Mask Generation]
  A7 --> A8[Masked Align]
  A8 --> A9[Masked Tilt Removal]
  A9 --> B1[Background Zeroing]
  end
  subgraph Grain Finding
  B1 --> B2[Lower Thresholding]
  B2 --> B3[Guassian Filtering]
  B3 --> B4[Tidy Edges]
  B3 --> D1[DNA Tracing]
  B4 --> B5[Preliminary Statistics]
  B5 --> B6[Size Thresholding]
  B6 --> C1[Label Regions]
  B6 --> D1
  end
  subgraph Grain Statistics
  C1 --> C2[Calculate Points]
  C2 --> C3[Calculate Edges]
  C2 --> C4[Calculate Centroid]
  C3 --> C5[Calculate Radius Statistics]
  C3 --> C6[Convex Hull / Graham Scan]
  end

```

## DNA Tracing : Processing a single grain
