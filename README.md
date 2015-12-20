Experiments can be run via: python demo.py

The following external software is needed:
* Maxima
* Gmsh

The following Python packages are needed:
* numpy / scipy
* pymbolic
* modepy
* meshpy

The patch `meshpy.patch` needs to be applied to meshpy. It hacks
around an issue where meshpy does not return all the information
needed for constructing quadratic elements.
