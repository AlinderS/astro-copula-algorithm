# Copula algorithm
Automated algorithm for finding copulas for astronomical data, derived from https://github.com/aaryapatil/elicit-disk-copulas

Used in Alinder et. al. (2025) https://arxiv.org/abs/2511.10092 to derive a copula for separating stars in the [Mg/Fe] - [Fe/H] plane.

# Example run:
Expects data in the form of an Astropy table.
```
from copula import copula_split_finder

group_one_indices, group_two_indices, poly_line = copula_split_finder(data)
```

# Attribution
Alinder, Simon; Bensby, Thomas; McMillan, Paul. Impact of selection criteria on the structural parameters of the Galactic thin and thick discs https://arxiv.org/abs/2511.10092

