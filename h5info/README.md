# h5info

H5MD is a file format specification, based upon HDF5, aimed at the efficient and portable storage of molecular data (e.g. simulation trajectories, molecular structures, …).
Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed to store and organize large amounts of data.

HDF5 simplifies the file structure to include only two major types of object:
HDF Structure Example

    Datasets, which are typed multidimensional arrays
    Groups, which are container structures that can hold datasets and other groups

This results in a truly hierarchical, filesystem-like data format.
Metadata is stored in the form of user-defined, named attributes attached to groups and datasets.

The following .py script provides an alternative to h5dump command line tool, to display the content of an HDF5 file. 
