import torch
import os
from utils import get_project_path

result_directory = os.path.join(get_project_path(project_name="Defense"), "defense_result")

result_path = os.path.join(result_directory, "ModelB_MultilevelMemoryUnet_t=50.pth")

result = torch.load(result_path)

for key in result.keys():
    accs = result[key]
    print(f"-------------------------------------------------\n"
          f"test result with {key}:\n"
          f"NorAcc:{accs[0]:.3f}\n"
          f"RecAcc:{accs[1]:.3f}\n"
          f"AdvAcc:{accs[2]:.3f}\n"
          f"RAvAcc:{accs[3]:.3f}\n"
          f"-------------------------------------------------")