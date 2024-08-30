ID 230248461

This work is totally based on the framework: https://github.com/utiasDSL/gym-pybullet-drones/tree/main

Install

git clone https://github.com/FangZ53/FZDissertation.git

cd 1D Task

conda create -n drones python=3.10

conda activate drones

pip3 install --upgrade pip

pip3 install -e . # if needed, sudo apt install build-essential to install gcc and build pybullet

To see the result of 1D task:

cd gym_pybullet_drones/examples/

python3 video.py

To see the result of 3D task:

cd ...FZDissertation/3D Task

PID control:

python test_multiagent.py --exp ./results/save-leaderfollower-2-cc-kin-pid-08.25.2024_18.15.17

VEL control:

python test_multiagent.py --exp ./results/save-leaderfollower-2-cc-kin-vel-08.25.2024_17.13.24

RPM control:

python test_multiagent.py --exp ./results/save-leaderfollower-2-cc-kin-rpm-08.25.2024_16.22.30