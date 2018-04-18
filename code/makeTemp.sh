#!/bin/bash

g++ -I/afs/cs.cmu.edu/academic/class/15418-s18/public/opencv-418/include main.cpp -L/afs/cs.cmu.edu/academic/class/15418-s18/public/opencv-418/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_stitching -lopencv_imgproc -o stitch
