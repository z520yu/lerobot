# Create virtual environment
uv venv --python 3.10 examples/piper_effort/.venv
source examples/piper_effort/.venv/bin/activate
uv pip sync examples/piper_effort/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python -m examples.piper_effort.main