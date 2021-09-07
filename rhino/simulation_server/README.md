# Simulation Server for Rhino

Since physics simulations cannot easily be done within `Grasshopper`, we use a client-server solution to compute the stability of an aggregation. Specifically, we run a python-based simulation server independently of `Rhino` and a `Grasshopper` script simply sends simulation tasks via the network interface and receives the results on the same way. To avoid the massive overhead introduced by `Coppeliasim`, we use `pybullet` for the simulations.  

## Installation

The simulation server requires a working `python 3.6` or higher installation. The required packages can be installed via the command line, using the provided `requirements.txt` file:

`pip install -r requirements.txt`

## Usage

The following command runs the server:

`python /path/to/simulation_server.py`

Optionally, the `-v` command line option can be used to visualize the simulation:

`python /path/to/simulation_server.py -v`

To see a complete list of options, type 

`python /path/to/simulation_server.py -h`

After the server has been started, the `Grasshopper` script will connect to it via sockets and send a simulation request whenever the aggregation changes. As soon as the simulation is complete, the server will reply with the final poses of each part, which are then visualized in `Rhino`. For further details see `simple_aggregation_decomposed.gh`.

The server can be terminated by sending an interrupt signal to the executing console: `ctrl+c`.

