#!/bin/bash
cd /tmp
apt-get download libsm6 libice6 2>&1 | tail -3
mkdir -p /home/nachoglezmur/mujoco/.syslibs
for d in libsm6*.deb libice6*.deb; do
  [ -f "$d" ] && dpkg -x "$d" /home/nachoglezmur/mujoco/.syslibs && echo "extraido $d"
done
find /home/nachoglezmur/mujoco/.syslibs -name "*.so*"
echo ---
LD_LIBRARY_PATH=/home/nachoglezmur/mujoco/.syslibs/usr/lib/x86_64-linux-gnu ldd /home/nachoglezmur/mujoco/.venv/lib/python3.12/site-packages/cv2/qt/plugins/platforms/libqxcb.so 2>&1 | grep -i "not found" | head -20
echo LDD_DONE
