#!/bin/bash

sudo apt-get install cmake
curl -O http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar -zxvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config shared=1
sudo make install
export METIS_DLL=/usr/local/lib/libmetis.so
