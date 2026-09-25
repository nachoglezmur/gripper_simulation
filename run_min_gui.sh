#!/bin/bash
export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
export MUJOCO_GL=glfw
export PYTHONUNBUFFERED=1
timeout 25 /home/nachoglezmur/mujoco/.venv/bin/python -u /home/nachoglezmur/mujoco/gui_min_test.py
echo EXITCODE=$?
