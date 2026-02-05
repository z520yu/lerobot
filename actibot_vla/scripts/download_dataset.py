# 正确的LeRobot数据集加载下载方式
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import snapshot_download
# 使用预转换的LeRobot格式LIBERO数据集
# dataset = LeRobotDataset("physical-intelligence/libero")
# dataset = LeRobotDataset("gauravpradeep/t01_piper_pick_and_place_bimanual_lerobot")
# dataset = LeRobotDataset("breezewrf/PiperLeRobot_PPTape_TwoCam_Train_v2.1")
# dataset = LeRobotDataset("wego-hansu/piper_test")
dataset = LeRobotDataset("destroy314/agilex_push_button")

print(f"数据集信息: {dataset}")
print(f"数据集大小: {len(dataset)}")
print(f"特征: {dataset.features}")

# local_dir = snapshot_download(
#                 repo_id="destroy314/agilex_push_button", 
#                 repo_type="dataset",
#                 local_dir="/home/a/data-processing/dataset",
#                 resume_download=True,
#                 local_dir_use_symlinks=False)
# print(f"本地模型目录: {local_dir}")