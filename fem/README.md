# Finite-element model and gradient networks

This folder contains the FEMM model of the 2.6-kW six-phase bearingless synchronous reluctance machine, the generation of the dataset, and the training of the gradient networks. The results are in `examples/datasets/` and `examples/trained_models/`, and they are used by the examples. The model, the dataset, and the networks are described in the [documentation](https://aalto-electric-drives.github.io/motulator-bearingless/fem.html).

## Usage

The scripts need [femm-qt](https://github.com/dcm3c/femm-qt), the cross-platform FEMM 5.0, and its Python interface in the same environment, installed from the femm-qt checkout with `pip install -e pyfemm` (not the `pyfemm` package of FEMM 4.2), as well as the optional dependencies, `pip install -e .[fem]`. The Python interface finds `femm_headless` in the build directory of the checkout, in the directory given by `FEMM_DIR`, or on `PATH`.

```bash
python sample_plan.py plan.npz
python run_maps.py plan.npz results.txt 7          # Continues an interrupted run
python make_dataset.py plan.npz results.txt ../examples/datasets/bsyrm_fem.npz
python check_model.py boundary mesh symmetry hessian field field.png

M=../examples/trained_models
D=../examples/datasets/bsyrm_fem.npz
python train_gradnet.py $D $M/bsyrm_fem_curr_map_harm_softmax_d128.npz --map current --embed 128 --k 6 36 --seed 4
python train_gradnet.py $D $M/bsyrm_fem_curr_map_softmax_d64.npz --map current --embed 64
python train_gradnet.py $D $M/bsyrm_fem_flux_map_softmax_d128.npz --map flux --embed 128 --s_2 0.007
python evaluate_gradnet.py $D $M/bsyrm_fem_curr_map_harm_softmax_d128.npz
python compare_control.py results.json
```

More points can be appended to the plan: `run_maps.py` skips the points already solved. The temporary files of FEMM go to `/dev/shm/bsyrm_femm` (or set `BSYRM_FEMM_WORK`).

| File                                               | Contents                                                          |
| -------------------------------------------------- | ----------------------------------------------------------------- |
| `bsyrm_femm.py`                                    | Parametric geometry, materials, winding, and the solver interface |
| `sample_plan.py`, `run_maps.py`, `make_dataset.py` | Dataset                                                           |
| `check_model.py`, `field_plot.py`                  | Checks of the model and field plots                               |
| `train_gradnet.py`, `evaluate_gradnet.py`          | Training of the networks and their errors                         |
| `compare_control.py`                               | Direct force control with different machine and controller models |
