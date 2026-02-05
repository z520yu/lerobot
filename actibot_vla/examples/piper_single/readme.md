终端窗口 1：

```bash
# Create virtual environment
uv venv --python 3.10 examples/piper_single/.venv
source examples/piper_single/.venv/bin/activate
uv pip sync examples/piper_single/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python -m examples.piper_single.main
```

终端窗口 2：

```bash
roslaunch piper_single ros_nodes.launch
```

终端窗口 3：

```bash
uv run scripts/serve_policy.py --env piper_single
```