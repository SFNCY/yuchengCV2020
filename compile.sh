#!/bin/bash

sudo rm -rf ./build/*
cd build
cmake ../
make
./TestApplication