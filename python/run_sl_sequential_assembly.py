import json
import traceback
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import trimesh

from aggregation import TemplatePart, Transformation
from sequential_assembly.placing import *
from sequential_assembly.sequential_assembly_task import SequentialAssemblyTask
from sequential_assembly.sl_sequential_assembly import SLSequentialAssembly, PlanningFailedException, \
    PlacingFailedException
from sequential_assembly.construction_plan_parser_rhino import ConstructionPlanParserRhino

parser = ArgumentParser("Runs the sequential assembly with SL blocks.")
parser.add_argument("mesh_path", type=str)
parser.add_argument("task_path", type=str)
parser.add_argument("--real-robot-ip", type=str,
                    help="The IP of the real robot to execute the trajectories on, without an IP the trajectories are"
                         "executed only in PyBullet.")
parser.add_argument("--scene_config_path", type=str, default=None,
                    help="The path of the scene config json file.")
parser.add_argument("--calibrated_offsets_path", type=str, default=None,
                    help="The path of the calibrated scene config json file.")
parser.add_argument("--use-simple-placing-controller", action="store_true",
                    help="Whether to use the simple controller that just moves down for placing")
args = parser.parse_args()

mesh_path = Path(args.mesh_path)
task_path = Path(args.task_path)
if args.scene_config_path is None:
    scene_config_path = Path(__file__).parent.parent / "config" / "sl_scene_config.json"
else:
    scene_config_path = Path(args.scene_config_path)
if args.calibrated_offsets_path is None:
    calibrated_offsets_path = None
else:
    calibrated_offsets_path = Path(args.calibrated_offsets_path)

sub_meshes = []
for mesh_file in (mesh_path / "decomposed").iterdir():
    with mesh_file.open() as f:
        sub_meshes.append(trimesh.load(f, file_type="obj").apply_scale(1e-3))
with (mesh_path / "sl.obj").open() as f:  # TODO: Name should be generic
    complete_mesh = trimesh.load(f, file_type="obj").apply_scale(1e-3)

sl_template_part = TemplatePart("sl_block", [], sub_meshes, complete_mesh, 0.2, 1.0)  # TODO: Mass should be passed

construction_plan_parser = ConstructionPlanParserRhino(sl_template_part.mesh)

if calibrated_offsets_path is not None:
    with calibrated_offsets_path.open() as calibrated_offsets_file:
        calibrated_offsets = json.load(calibrated_offsets_file)
    pickup_table_offset_3d = np.concatenate((calibrated_offsets["pickup_table"], np.zeros(1)))
    place_table_offset_3d = np.concatenate((calibrated_offsets["place_table"], np.zeros(1)))
else:
    pickup_table_offset_3d = np.zeros(3)
    place_table_offset_3d = np.zeros(3)
pickup_table_transformation = Transformation(pickup_table_offset_3d + np.array([0, -0.0055, 0])) # TODO: Where does the second offset come from?
place_table_transformation = Transformation.from_pos_euler(np.array([-1.7, 0, 0]) + place_table_offset_3d, [0, 0, 0])
spawn_poses, goal_poses = \
    construction_plan_parser.parse(task_path, pickup_table_transformation, place_table_transformation)

if args.use_simple_placing_controller:
    placing_controller = SimplePlacingController()
else:
    placing_controller = MagicPlacingController()


sequential_assembly = SLSequentialAssembly(sl_template_part, scene_config_path, placing_controller,
                                           calibrated_offsets_path=calibrated_offsets_path,
                                           real_robot_ip=args.real_robot_ip)
sequential_assembly.initialize()

task = SequentialAssemblyTask(spawn_poses, goal_poses)

while True:
    try:
        sequential_assembly.solve_task(task)
    except (PlanningFailedException, PlacingFailedException) as e:
        traceback.print_exc()
