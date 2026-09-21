# Infeasibility certificates for "The Laplacian S_{n,n} conjecture is true"
Supplementary material for the paper "The Laplacian S_{n,n} conjecture is true" by Nathaniel Johnston.

This repo contains the infeasibility certificates that are used in the proof of Theorem 3.7 (in particular, to prove that there does not exist a simple graph with Laplacian spectrum equal to S_{n,n} when n equals 16, 20, 21, 24, 25, or 28). Python 3.10 or later is required.

The certificates are contained in the six .zip files. To verify them, upzip them all so that a single directory contains snn_verify.py as well as six subdirectories called n16, n20, n21, n24, n25, and n28. The verifier can then be run for a specific order via "python snn_verify.py n24", for example.
