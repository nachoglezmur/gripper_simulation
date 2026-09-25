#!/bin/bash
export DISPLAY=:0
unset WAYLAND_DISPLAY
export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
export MUJOCO_GL=glfw
export PYTHONUNBUFFERED=1
timeout 20 /home/nachoglezmur/mujoco/.venv/bin/python -u /home/nachoglezmur/mujoco/gui_min_test.py
echo EXITCODE=$?
