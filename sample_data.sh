#!/bin/bash
g++ data_sampler.cpp -o data_dampler `pkg-config opencv --libs`
./data_dampler ./sampling/rgb ./sampling/mask