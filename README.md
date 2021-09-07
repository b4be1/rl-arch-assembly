## Installation
#### Pull all submodules
`git submodule update --init`

#### Install dependencies
Install the following libraries:
* `openscad` (for `trimesh`)
* `fcl` (for `python-fcl`)
* `libspatialindex` (for `rtree`)

Install a patched version of PyBullet (this is needed since there is currently a bug in the official version of PyBullet's UrdfEditor).
```
cd /path/to/this/repo/submodule/bullet3
cmake .
make
pip install .
```

Install a patched version of Trimesh (this is needed because of a bug with the blender backend):

`pip install /path/to/this/repo/submodule/trimesh`


In case your `fcl` version is `0.6` or higher, you need to install a patched version of `python-fcl`:

`pip install /path/to/this/repo/submodule/python-fcl`

Otherwise, take the current `pypi` version:<br>
`pip install python-fcl`

Then the remaining packages can be installed via

`pip install -r requirements.txt`

Install assembly-gym (you need to make sure that you are on the branch `feature/simulator-abstraction`)
```
cd /path/to/this/repo/submodule/assembly-gym
git checkout feature/simulator-abstraction
pip install .
```


Our simulation of the Digit sensor uses `cupy` for GPU acceleration. While the simulation can be done without `cupy`, it will be significantly slower without. If you want to use `cupy`, install it with

`pip install cupy`

## Structure of the project
The project is structured in the following packages
* `aggregation`: &emsp; Classes to generate random stable structures to be used as training data for the placing controller
* `algorithm`: &emsp; Reinforcement learning algorithms that are used to learn the placing controller
* `gripper`: &emsp; The interface to the real Schunk gripper
* `scene`: &emsp; Constructs a general simulated assembly scene (with tables, a robot, and blocks)
* `sensors`: &emsp; An implementation of a simulation of the DIGIT sensor
* `sequential_assembly`: &emsp; The planning-based solver for the problem of sequentially grasping and transporting blocks close to their target location
* `task`: &emsp; Classes for defining the gym environment to train the placing controller on
    * `controllers`: &emsp; Controllers define the semantics of the actions of the agent (e.g. torques, target joint velocities)
    * `rewards`: &emsp; Define the rewards of the environment
    * `sensors`: &emsp; Sensors define the observation that the agent get
    * `wrappers`: &emsp; Wrappers that modify the shape of the actions and obsreations (e.g. flatten dictionaries) so that the task can be used with stable-baselines3 agents
    
## Execution
### Generate training data for the placing controller:
```
mkdir -p data/tasks
cd python
python generate_tasks.py ../tasks/box_double_short ../example/rhino_export/box --min-parts 1 --max-parts 2 --parts short -n 10000 --min-parts-to-place 1 --max-parts-to-place 1
```

This will generate 10000 stable structures consisting of two part of which the robot has to place one. 
The result is stored in `data/tasks/box_double_short`.
Add the `--view-structures` option to visualize the structures.

### Train a placing controller with TD3
```
cd python
python run_training.py ../config/run_configs/td3_stacking_double.py
```

This command required that training data is provided at `data/tasks/box_double_short` (see above to generate the training data).

### Run the planning-based controller for grasping and transporting
```
cd python
python run_sl_sequential_assembly.py ../example/meshes/sl ../example/sequential_assembly/place_from_top
```