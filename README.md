# PyCHAMELEON

Python 2.7 implementation of the clustering algorithm CHAMELEON[1].

Depends on METIS for Python.

Consider checking out https://github.com/Moonpuck/chameleon_cluster for an improved Python3 version.

## Installing (MacOS instructions)

1. Install requirements.

```
pip install -r requirements.txt
```

2. Install METIS

```
brew install cmake
curl -O http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar -zxvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config shared=1
make install
export METIS_DLL=/usr/local/lib/libmetis.dylib
```

3. Run sample code

```
python -i main.py
```



## References

[1] Karypis, George, Eui-Hong Han, and Vipin Kumar. "Chameleon: Hierarchical clustering using dynamic modeling." *Computer* 32.8 (1999): 68-75.
http://ieeexplore.ieee.org/abstract/document/781637/
