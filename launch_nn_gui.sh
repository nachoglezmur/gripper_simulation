#!/bin/bash
export DISPLAY=:0
unset WAYLAND_DISPLAY
export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
export MUJOCO_GL=glfw
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export LD_LIBRARY_PATH=/home/nachoglezmur/mujoco/.syslibs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
cd /home/nachoglezmur/mujoco
nohup /home/nachoglezmur/mujoco/.venv/bin/python -u -m sim.cdpr_nn_gui > /tmp/cdpr_nn_gui.log 2>&1 &
echo LAUNCHED_PID=$!
sleep 25
echo ---LOG---
cat /tmp/cdpr_nn_gui.log
echo ---PS---
ps aux | grep cdpr_nn_gui | grep -v grep
