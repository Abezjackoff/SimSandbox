# SimSandbox

---

## About
This project contains a collection of algorithms
and a sandbox for simulation of dynamic and control systems in Python

---

## Usage

1. Navigate to the folder where you would like to keep the project.
2. Clone the repository: \
`git clone https://github.com/Abezjackoff/SimSandbox.git`
3. Install necessary Python packages: \
`python -m pip install -r requirements.txt`
4. Explore the project files, then import classes or methods that 
interest you into `sandbox.py`, build a system for simulation and run.
There are some examples already prepared.

---

## File Structure

### Control
This folder contains control algorithms and base classes to create and
execute controllers such as LQR or MPC. Some controllers have interfaces
to connect with a plant.

### Dynamics
This folder contains models of dynamic systems that could be simulated
stand-alone or included in a control loop.

### Resources
This folder contains any data or configuration files that might be
needed for simulation or tuning the models.
