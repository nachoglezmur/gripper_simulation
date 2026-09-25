#!/bin/bash
echo ---LOG---
cat /tmp/mujoco_gui.log
echo ---PS---
ps aux | grep cable_robot_gui | grep -v grep
echo --- ls WSLg ---
ls /tmp/.X11-unix/
echo DISPLAY=$DISPLAY WAYLAND=$WAYLAND_DISPLAY
