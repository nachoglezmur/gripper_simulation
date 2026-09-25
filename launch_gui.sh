#!/bin/bash
export DISPLAY=:0
unset WAYLAND_DISPLAY
export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
export MUJOCO_GL=glfw
export PYTHONUNBUFFERED=1
nohup /home/nachoglezmur/mujoco/.venv/bin/python -u /home/nachoglezmur/mujoco/cable_robot_gui.py > /tmp/mujoco_gui.log 2>&1 &
echo LAUNCHED_PID=$!
sleep 6
echo ---LOG---
cat /tmp/mujoco_gui.log
echo ---PS---
ps aux | grep cable_robot_gui | grep -v grep
