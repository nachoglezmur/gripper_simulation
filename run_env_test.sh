#!/bin/bash
export JAX_PLATFORMS=cuda
/home/nachoglezmur/mujoco/.venv/bin/python -m pytest tests/test_env.py::test_gpu_stacks -v -s
echo EXITCODE=$?
