#!/usr/bin/env python3

from pathlib import Path

if __name__ == "__main__":
    directory = Path(__file__).parent
    print("Directory: {}".format(directory))

    failed = []

    for p in directory.glob('**/*'):
        imp = False
        if p.is_dir():
            if (p / "__init__.py").exists():
                imp = True
        elif p.suffix == ".py" and p.stem != "__init__":
            imp = True

        if imp:
            relative_path = str(p.relative_to(directory))
            if relative_path.endswith(".py"):
                relative_path = relative_path[:-3]
            module_name = relative_path.replace("/", ".")
            print("Importing {}... ".format(module_name))
            try:
                __import__(module_name)
                print("  Ok")
            except ImportError as e:
                failed.append(module_name)
                print("  Failed")
                print(e)

    if len(failed) == 0:
        print("All modules imported successfully.")
    else:
        print("The following modules could not be imported:")
        for m in sorted(failed):
            print("  {}".format(m))

print("")
import torch

print("Found {} CUDA devices".format(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print("  {}".format(torch.cuda.get_device_name(0)))