import argparse
import os
import pickle
import shutil
import socket
import struct
import time
from tempfile import TemporaryDirectory
from typing import List, Optional, Any

import pybullet as p
import numpy as np
import pybullet_data
from scipy.spatial.transform import Rotation

VERSION = (1, 0, 0, 0)


class Connection:
    def __init__(self, s: socket):
        self._socket = s
        self._buffer = bytearray()

    def receive_object(self) -> Optional[Any]:
        while len(self._buffer) < 4 or len(self._buffer) < struct.unpack("<L", self._buffer[:4])[0] + 4:
            new_bytes = self._socket.recv(16)
            if len(new_bytes) == 0:
                return None
            self._buffer += new_bytes

        length = struct.unpack("<L", self._buffer[:4])[0]
        header, body = self._buffer[:4], self._buffer[4:length + 4]

        obj = pickle.loads(body)

        self._buffer = self._buffer[length + 4:]

        return obj

    def send_object(self, d: Any):
        body = pickle.dumps(d, protocol=2)
        header = struct.pack("<L", len(body))
        msg = header + body
        self._socket.send(msg)


def simulate(parts: List, sim_length_s: float = 10.0, time_step_s: float = 1 / 240,
             real_time_factor: Optional[float] = None) -> List[List[float]]:
    print("Simulating...")
    min_z_pos = np.infty
    bodies = [None] * len(parts)
    for i, part in enumerate(parts):
        trans = part["transformation"]
        h = part["hash"]
        mass = part["mass_kg"] if "mass_kg" in part else 1
        trans = np.array(trans) * 0.001
        pos = trans[:3, 3]
        rotation = Rotation.from_matrix(trans[:3, :3])
        bodies[i] = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=mesh_store[h], basePosition=pos,
                                      baseOrientation=rotation.as_quat())
        dynamics_kwargs = part.get("dynamics", {})

        if len(dynamics_kwargs) > 0:
            p.changeDynamics(bodies[i], linkIndex=-1, **dynamics_kwargs)
        min_pos, max_pos = p.getAABB(bodies[i])
        min_z_pos = np.minimum(min_pos[2], min_z_pos)

    p.resetBasePositionAndOrientation(plane, [0, 0, min_z_pos - 0.003], [0.0, 0.0, 0.0, 1.0])
    p.setTimeStep(time_step_s)

    next_step = time.time()
    for i in range(int(sim_length_s / time_step_s)):
        if real_time_factor is not None:
            time.sleep(max(next_step - time.time(), 0))
            next_step += time_step_s / real_time_factor
        p.stepSimulation()

    # Obtain new positions of parts and return them
    new_poses = [p.getBasePositionAndOrientation(b) for b in bodies]

    new_rotation_matrices = np.array([Rotation.from_quat(q).as_matrix() for t, q in new_poses])
    new_translations = np.array([t for t, q in new_poses])

    new_transforms = np.tile(np.eye(4)[np.newaxis], reps=(len(new_poses), 1, 1))
    new_transforms[:, :3, 3] = new_translations * 1000
    new_transforms[:, :3, :3] = new_rotation_matrices

    # Cleanup
    for b in bodies:
        p.removeBody(b)

    print("Done")

    return new_transforms.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregation Simulation Server")
    parser.add_argument("-e", "--expose", action="store_true", help="Expose this server to the network "
                                                                    "(DO NOT ENABLE IN PUBLIC NETWORKS).")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port this server listens to.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize the simulation with pybullet's built-"
                                                                       "in visualizer.")
    parser.add_argument("-r", "--real-time-factor", type=float,
                        help="Real time speed factor of the simulation. Setting 0.5 here means that the simulation runs"
                             " in 50% of real time. Can be overwritten from Grasshopper. This option is ignored if -v"
                             " is not set. Default: no real time factor (simulate as fast as possible).")
    args = parser.parse_args()

    version_st = tuple(map(str, VERSION))
    print("Simulation Server v. {}".format(".".join(version_st)))

    active_connections = {}

    mesh_store = {}

    physicsClient = p.connect(p.GUI if args.visualize else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])

    world_state = p.saveState()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0" if args.expose else "localhost", args.port))
        s.listen(5)
        s.settimeout(0)

        print("Listening to port {}...".format(args.port))

        with TemporaryDirectory() as tmpdir:
            while True:
                try:
                    clientsocket, address = s.accept()
                    print("Client connected from {}:{}".format(*address))
                    clientsocket.settimeout(0)
                    c = Connection(clientsocket)
                    active_connections[clientsocket] = (c, time.time())
                except BlockingIOError:
                    pass

                delete_list = []
                now = time.time()
                for clientsocket, (conn, t) in active_connections.items():
                    try:
                        m = conn.receive_object()
                    except (BlockingIOError, ConnectionResetError):
                        m = None
                    if m is not None:
                        if "version" not in m or m["version"][:-2] != VERSION[:-2]:
                            error = "Received request with mismatched version (v. {} while my version is {}). " \
                                    "Please make sure that you are running matching versions of the simulation server " \
                                    "and the Grasshopper client (either both {}.x.x or both {}.x.x).".format(
                                ".".join(map(str, m["version"])), ".".join(version_st), ".".join(version_st[:-2]),
                                ".".join(map(str, m["version"][:-2])))
                            print(error)
                            conn.send_object(
                                {"type": "error", "err": "VERSION_MISSMATCH", "desc": error, "version": VERSION})
                        elif m["type"] == "sim":
                            # Check if all meshes are present
                            mesh_hashes = set(part["hash"] for part in m["parts"])
                            unknown_hashes = [m for m in mesh_hashes if not m in mesh_store]
                            if len(unknown_hashes) > 0:
                                conn.send_object({"type": "mesh_request", "hashes": mesh_hashes, "version": VERSION})
                            else:
                                # Simulate
                                d = {
                                    "sim_length_s": 5.0,
                                    "time_step_s": 1 / 240
                                }
                                d.update(m)
                                if args.visualize:
                                    rtf = args.real_time_factor
                                    if "real_time_factor" in d:
                                        rtf = d["real_time_factor"]
                                else:
                                    rtf = None
                                new_poses = simulate(d["parts"], d["sim_length_s"], d["time_step_s"], rtf)
                                conn.send_object(
                                    {"type": "sim_result", "id": m["id"], "new_poses": new_poses, "version": VERSION})
                        elif m["type"] == "meshes":
                            for h, mesh_lst in m["meshes"].items():
                                if not h in mesh_store:
                                    d = os.path.join(tmpdir, h)
                                    if os.path.exists(d):
                                        shutil.rmtree(d)
                                    os.mkdir(d)
                                    for i, mesh in enumerate(mesh_lst):
                                        with open(os.path.join(d, "{}.obj".format(i)), "w") as f:
                                            f.write("o {}_{}\n".format(h, i))
                                            for v in mesh["vertices"]:
                                                f.write("v {:0.8f} {:0.8f} {:0.8f}\n".format(*(np.array(v) * 0.001)))
                                            for fa in mesh["faces"]:
                                                f.write("f {} {} {}\n".format(*(np.array(fa) + 1)))
                                    mesh_store[h] = p.createCollisionShapeArray(
                                            [p.GEOM_MESH] * len(mesh_lst),
                                            fileNames=[os.path.join(d, "{}.obj".format(i)) for i in
                                                       range(len(mesh_lst))])

                                # In case that ever gets fixed
                                # mesh_store[h] = p.createCollisionShapeArray(
                                #     [p.GEOM_MESH] * len(mesh_lst),
                                #     vertices=[[[ei * 0.001 for ei in e] for e in m["vertices"]] for m in mesh_lst],
                                #     indices=[[ei for e in m["faces"] for ei in e] for m in mesh_lst])
                        else:
                            print("Unknown message type \"{}\"".format(m["type"]))

                    if now - t > 60:
                        delete_list.append(clientsocket)

                for cs in delete_list:
                    cs.close()
                    del active_connections[cs]

                time.sleep(0.1)
    finally:
        try:
            s.shutdown(socket.SHUT_RDWR)
        finally:
            s.close()

