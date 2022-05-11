## Parameter Configuration

*todo: document YAML config.*

The parameters that the software uses for analysis of the data can be configured in `Config.ini` by simply opening the file in a text editor and changing the variables. You do not need to edit the code to change the parameters.

When updating TopoStats, the `Config.ini` file is ignored, so your parameters are maintained. Different sets of parameters can be saved for different sample types in the config file, and the sections for the different sample types are labelled in square brackets.

If no config file is found while running TopoStats, it will make a copy of the default config file.