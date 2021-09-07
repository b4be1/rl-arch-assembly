import argparse
from pathlib import Path

import numpy as np


def get_new_idx(old_idx):
    return ((int(old_idx) - 1) % 8) + 1

def extract_coordinates(line):
    vertex_str = line[1:].split()
    assert len(vertex_str) == 3
    return np.array([float(c) for c in vertex_str])


parser = argparse.ArgumentParser(description="Centers submesh files")
parser.add_argument("mesh_dir", type=str)
parser.add_argument("x", type=float)
parser.add_argument("y", type=float)
parser.add_argument("z", type=float)
args = parser.parse_args()

mesh_dir = Path(args.mesh_dir)
offset = np.array([args.x, args.y, args.z])

for file in mesh_dir.iterdir():
    if file.is_file():
        with file.open("r") as f:
            new_lines = []
            for line in f.readlines():
                if line[:2] == "v ":
                    vertex = extract_coordinates(line)
                    new_vertex = vertex + offset
                    new_line = " ".join(["v"] + [str(c) for c in new_vertex]) + "\n"
                    pass
                else:
                    new_line = line
                new_lines.append(new_line)

        with file.open("w") as f:
            f.writelines(new_lines)
