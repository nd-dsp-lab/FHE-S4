
# About
This outlines how to run a shorted version of our S4 approximation in C++ (EvalChebyshevSeries_demo.cpp), up until post-activation, where we can see the reults of EvalChebyshevSeries.

In EvalChebyshevSeries_demo.cpp...
See code starting at line 294 for the use of EvalChebyshevSeries in our approximation
See code starting at line 145 for OpenFHE context

The current version implements a run of EvalChebyshev where the first coefficient is divided by 2 (see lines 305-307). The output shows minimal error.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/nd-dsp-lab/FHE-S4
cd FHE-S4
```

### 2. Set up Virtual Environment
Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Download Python Dependencies

```bash
./scripts/setup_python_venv.sh
```
Ignore the instructions at the end.


### 4. Export json from Python
```bash
python export_for_openfhe_new.py
```


### 5. Make C++ scripts
```bash
cd build
cmake .. -DOpenFHE_DIR=/usr/local/lib/OpenFHE
make -j"$(nproc)"
```

### 6. Run the Script
```bash
cd ..
./build/eval_chebyshev_series
```
It may take a few seconds to run to completion