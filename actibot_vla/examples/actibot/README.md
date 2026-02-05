```bash
# Create virtual environment
cd ~/pi05
uv venv --python 3.10 examples/actibot/.venv
source examples/actibot/.venv/bin/activate
uv pip sync examples/actibot/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python -m examples.actibot.main
```