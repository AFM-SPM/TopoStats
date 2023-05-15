# Knot Detangling

Often there are molecules intertwinned with each other and it is desirable to determine how these are interacting.

The flow diagram below shows the steps implemented on the
[maxgamill-sheffield/cats](https://github.com/AFM-SPM/TopoStats/tree/maxgamill-sheffield/cats) branch. This is subject
to refactoring and may change in the future but serves as a reference when undertaking such work.

```{mermaid}
%%{init: {'theme': 'base',
         }
}%%
graph TD;

  subgraph "Class : dnaTrace() Runs with dnaTrace.trace_dna()"
  A1(["Method: trace_dna()\nINPUT: Grain\nImage\n Skeleton\n"]) --> A2(["Action : Smooth Grains\nDescription : Performs either ginary dilation or Gaussian Smoothing\nMethod : smooth_grains()"])
  D4 --> A4(["Itreate over ALL grains\nSkeletonise : getSkeleton()"])
  A4 --> A5(["Action : Remove Noise AKA\nMethod : purge_obvious_crap()"])
  A5 --> A6(["Action : Determine Morphology on Disordered Traces\n(Linear or Circular)"])
  A6 --> A7(["Action : Order Traces"])
  A7 --> A8(["Action : Determine Morphology on Ordered Traces\n(Linear or Circular)"])
  A8 --> A9(["Get Fitted Trace aligns trace to backbone of DNA"])
  A9 --> A10(["Get Splined Trace"])
  A10 --> A11(["Meaure  : Contour Length"])
  A11 --> A12(["Measure : End to End Distance"])
  A12 --> A13(["Report Basic Stats"])
  end
  subgraph "Class : nodeStats main function is nodeStats.analyse_nodes()"
  A4 --> B2(["Action : CALCULATION\nDescription : Convolution on Skeleton calculates number of nearest neighbours for each pixel in the skeleton\nMethod : convlve_skelly()\nLines : "])
  B2 --> B3(["Action : CHECK\nDescription : if there are nodes this is where there are overlaps in the skeleton from twists or overlapping molecules\nLines : "])
  B3 --> B4(["Action : CHECK\nDescription : if there are nodes Connect Close Nodes - if there two nodes in a region\nMethod : connect_close_nodes()\nLines : "])
  B4 --> B5(["Action : CALCULATION\nDescription : Highlight node centres\nDescription :  Gets the highest point out of a node from a skeleton\nMethod highlight_node_centres()\nLines : "])
  end
  subgraph "Method : nodeStats.analyse_nodes()"
  B5 --> C1(["Action : CALCULATE\nDescription : Box length - the area around a node that is looked at 20nm\nLines :"])
  C1 --> C2(["Action : CHECK\nDescription : Regions (of nodes) should be > 10pixels\nLines :"])
  C2 --> C3(["Action : CHECK\nDescription : Ensure average trace resides within the mask"\nLines :])
  C3 --> C4(["Action : CALCULATE\nDescription : Iterate over nodes : get local area of image and node\nLines :"])
  C4 --> C5(["Action : CALCULATE\nDescription : Dichotomise image as skeleton passed in had 2 for ends and 3 for nodes\nLines :"])
  C5 --> C6(["Action : CALCULATE\nDescription : Calculate the centres of nodes\nLines :"])
  C6 --> C7(["Action : CALCULATE\nDescriptoin : Get mask where centre has been removed\nLines :"])
  C7 --> C8(["Action : CALCULATE\nDescription : Label branches, these are the parts of the skeleton that come out of the node, could have >3\nLines :"])
  C8 --> C9(["Action : CHECK\nDescription : Check resolution is sufficient\nLines : "])
  C9 --> C10(["Action : CALCULATE\nDescription : For each branch in the array take a branch and order coordinates\nMethod : order_branch()"])
  C10 --> C11(["Action : CALCULATE \nDescription : Identify vectors\nFunction get_vector()\nLines :"])
  C11 --> C12(["Action : CALCULATE \nDescription : Pair vectors, these are almost contiguous branches which run together\nFunction : pair_vectors()\nLines :"])
  C12 --> C13(["Action : CALCULATE\nDescription : Join paired vectors, iterate through branch pairs\nLines :"])
  C13 --> C14(["Action : CALCULATE\nDescription : Order each branch\nFunction order_branches()\nLines :"])
  C14 --> C15(["Action : CALCULATE\nDescription : Linear interpolation by taking a skeletonised line between co-ordinates fills in the line through the node which has already been removed\nLines :"])
  C15 --> C16(["Action : CALCULATE\nDescription : Remove duplicate crossing co-ordinates\nLines :"])
  C16 --> C17(["Action : CALCULATE\nDescription : Combine branches, and interpolated crossing\nLines :"])
  C17 --> C18(["Action : CALCULATE\nDescription : Skeletonise again to ensure there are no branches.\nLines :"])
  C18 --> C19(["Action : CALCULATE\nDescription : Convert to image wide coordinates\nLines :"])
  C19 --> C20(["Action : CALCULATE\nDescription : Obtain heights and distances of branches\nFucntion : average_height_trace()\nLines :"])
  C20 --> C21(["Action : CALCULATE\nDescription : Identify over/under using full width half height (fwhm)\nFunction : fwhm2()\nLines :"])
  C21 --> C22(["Action : CALCULATE\nDescription : Plot the highest fwhm on top\nLines :"])
  C22 --> C23(["Action : CALCULATE\nDescription : Put unpaired branches back on image if any exist\nLines :"])
  C23 --> C24(["Action : CALCULATE\nDescription : Identify crossing angles from full branch vectors.\nLines :"])
  C24 --> C25(["Action : CALCULATE\nDescription : Save everything in a dictionary.\nLines :"])
  C25 --> C26(["Action : CALCULATE\nDescription : Ensure linear interpolation and included in the main image.\nLines :"])
  C26 --> C27(["Action : \nDescription : Make dictionary a property\nLines :"])
  C27 --> C28(["Action : CALCULATE\nDescription : Order Co-ordinates\nLines : "])
  C28 --> C29(["Action : CHECK\nDescription : Check there are no errors (WHAT MIGHT THESE BE?)\nLines : "])
  end
  subgraph "Description : Smooths grains Method :dnaTrace.smooth_grains()"
  A2 --> D1(["Action : CALCULATE\nDescription : Dilate grain"])
  A2 --> D2(["Action : CALCULATE\nDescription : Gaussian Smooth"])
  D1 --> D3(["Action : CHECK\nDescription : Which method gives smallest changes in number of pixels?"])
  D2 --> D3(["Action : CHECK\nDescription : Which method gives smallest changes in number of pixels?"])
  D3 --> D4(["Return smallest"])
  end
  subgraph "Description : Compile Trace Method : compile_trace()"
  C29 --> E1(["Action : INPUT\nDescription : Extract node coordinates, area_box corssing_cords, crossing_heights, crossing_distances"])
  E1 --> E2(["Action : EXTRACTION\nDescription : Extract nodes\nLines :"])
  E2 --> E3(["Action : CALCULATION \nDescription : Extract minus image (i.e. without nodes where traces cross)\nFunction : get_minus_img()\nLines : "])
  E3 --> E4(["Action : CALCULATION\nDescription : Image with only nodes (i.e. where traces cross)\nFunction : get_crossing_img()\nLines : "])
  E4 --> E5(["Action : CALCULATION\nDescription : Combine minus and cross/node images\nFunction : get_both_img()\nLines : "])
  E5 --> E6(["Action : CALCULATION\nDescription : Order the coordinates within the segments of the minus branches (crossing are already ordered)\nLines : "])
  E6 --> E7(["Action : \nDescription : Trace the molecule (BETTER DESCRIPTION?)\nFunction trace_mol()\nLines : "])
  E7 --> E8(["Action : \nDescription : (BETTER DESCRIPTION?)\nFunction get_visual_img()\nLines : "])
  E8 --> E9(["Action : \nDescription : Generate necessary data for plainar\nFunction get_pds() plainar diagrams\nLines : "])
  end
  subgraph "Description : Method : trace_mol()"
  E9 --> F1(["Pick one segment as start"])
  F1 --> F2(["Action : CALCULATION\nDescription : Remove segment and find end coordinates\nLines : "])
  F2 --> F3(["Action : CALCULATION\nDescription : Look at local area within image to find the next index.\nLines : "])
  F3 --> F4(["Action : CALCULATION\nDescription : Iterate over segments\nLines : "])
  F4 --> F5(["Action : CALCULATION\nDescription : Make sure the end of a coordinate trace matches the start of a segment\nLines : "])
  F5 --> F6(["Action : CALCULATION\nDescription : Separate molecules\nLines : "])
  F6 --> F7(["Action : CALCULATION\nDescription : Return molecule ordered co-ordinates could be many.\nLines : "])
  end
  subgraph "Description : Plotting Method : get_visual_img()"
  F7 --> G1(["Action : INPUT\nDescription : co-ordinate trace, full width half maximums, crossing coordinates"])
  G1 --> G2(["Action : ITERATE\nDescription : Iterate over molecule number and their coordinates\nLines : "])
  G2 --> G3(["Action : CALCULATION\nDescription : Add in the co-ordinates to a blank image as a separate colour i.e. unique numbers for each molecule identified\nLines : "])
  G3 --> G4(["Action : CHECK\nDescription : If more than one molecule there are nodes/overlaps there are crossing coordinates so ensure that over and under are the correct colours\nLines : "])
  G4 --> G5(["Action : CHECK\nDescription : If single molecule still want to know how branches are overlaid on each other, only crossing have colour on the upper region at present, at the moment the lower region should have colour but doesn't\nLines : "])
  end
  subgraph "Description : Find indices of the branch Method : get_pds()"
  G5 --> H1(["Action : INPU\nDescription : Co-ordinate trace, node center coordinates, full width half maximum, crossing co-ordinates"])
  H1 --> H2(["Sorry, I kind of lost the thread a bit at this point! FILL ME IN"])
  end
  style A1 fill:#648FFF,stroke:#999999
  style A2 fill:#648FFF,stroke:#000000
  style A4 fill:#648FFF,stroke:#000000
  style A5 fill:#648FFF,stroke:#000000
  style A6 fill:#648FFF,stroke:#000000
  style A7 fill:#648FFF,stroke:#000000
  style A8 fill:#648FFF,stroke:#000000
  style A9 fill:#648FFF,stroke:#000000
  style A10 fill:#648FFF,stroke:#000000
  style A11 fill:#648FFF,stroke:#000000
  style A12 fill:#648FFF,stroke:#000000
  style A13 fill:#648FFF,stroke:#000000
  style B2 fill:#DC267F,stroke:#000000
  style B3 fill:#DC267F,stroke:#000000
  style B4 fill:#DC267F,stroke:#000000
  style B5 fill:#DC267F,stroke:#000000
  style C1 fill:#AB274F,stroke:#000000
  style C2 fill:#AB274F,stroke:#000000
  style C3 fill:#AB274F,stroke:#000000
  style C4 fill:#AB274F,stroke:#000000
  style C5 fill:#AB274F,stroke:#000000
  style C6 fill:#AB274F,stroke:#000000
  style C7 fill:#AB274F,stroke:#000000
  style C8 fill:#AB274F,stroke:#000000
  style C9 fill:#AB274F,stroke:#000000
  style C10 fill:#AB274F,stroke:#000000
  style C11 fill:#AB274F,stroke:#000000
  style C12 fill:#AB274F,stroke:#000000
  style C11 fill:#AB274F,stroke:#000000
  style C12 fill:#AB274F,stroke:#000000
  style C13 fill:#AB274F,stroke:#000000
  style C14 fill:#AB274F,stroke:#000000
  style C15 fill:#AB274F,stroke:#000000
  style C16 fill:#AB274F,stroke:#000000
  style C17 fill:#AB274F,stroke:#000000
  style C18 fill:#AB274F,stroke:#000000
  style C19 fill:#AB274F,stroke:#000000
  style C20 fill:#AB274F,stroke:#000000
  style C21 fill:#AB274F,stroke:#000000
  style C22 fill:#AB274F,stroke:#000000
  style C23 fill:#AB274F,stroke:#000000
  style C24 fill:#AB274F,stroke:#000000
  style C25 fill:#AB274F,stroke:#000000
  style C26 fill:#AB274F,stroke:#000000
  style C27 fill:#AB274F,stroke:#000000
  style C28 fill:#AB274F,stroke:#000000
  style C29 fill:#AB274F,stroke:#000000
  style D1 fill:#328499,stroke:#000000
  style D2 fill:#328499,stroke:#000000
  style D3 fill:#328499,stroke:#000000
  style D4 fill:#328499,stroke:#000000
  style E1 fill:#008099,stroke:#000000
  style E2 fill:#008099,stroke:#000000
  style E3 fill:#008099,stroke:#000000
  style E4 fill:#008099,stroke:#000000
  style E5 fill:#008099,stroke:#000000
  style E6 fill:#008099,stroke:#000000
  style E7 fill:#008099,stroke:#000000
  style E8 fill:#008099,stroke:#000000
  style E9 fill:#008099,stroke:#000000
  style F1 fill:#6A4DFF,stroke:#000000
  style F2 fill:#6A4DFF,stroke:#000000
  style F3 fill:#6A4DFF,stroke:#000000
  style F4 fill:#6A4DFF,stroke:#000000
  style F5 fill:#6A4DFF,stroke:#000000
  style F6 fill:#6A4DFF,stroke:#000000
  style F7 fill:#6A4DFF,stroke:#000000
  style G1 fill:#AA00CC,stroke:#000000
  style G2 fill:#AA00CC,stroke:#000000
  style G3 fill:#AA00CC,stroke:#000000
  style G4 fill:#AA00CC,stroke:#000000
  style G5 fill:#AA00CC,stroke:#000000
  style H1 fill:#990000,stroke:#000000
  style H2 fill:#990000,stroke:#000000
```
