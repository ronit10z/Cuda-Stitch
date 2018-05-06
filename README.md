# Cuda Stitch - A GPU accelarated real-time video stitcher

## Summary

We implemented a program that stitches two video streams together with CUDA and compared its performance with a sequential implementation. We were able to get a 100x times speedup over the naive sequential version and a 5x speedup over the optimised sequential version. The GPU version is able to process two steams captured in 720p at 24 frames per-second.

## BACKGROUND

GPU accelarated video stitching.

To get started, clone the repository.

Install the following dependencies - 

1. CUDA version 8.0+
2. install 
