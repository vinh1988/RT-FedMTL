#!/bin/bash

# Project root (THIS is the important fix)
PROJECT_ROOT="/home/vinh/Documents/code/FedAvgLS"

# Experiment directory
WORK_DIR="$PROJECT_ROOT/experiment_new_solution/models/tiny_bert/fl-mtl-slms-berttiny-sts-qqp-sst2"

# Virtual environment
VENV_ACTIVATE="$PROJECT_ROOT/venv/bin/activate"

# Terminal 1: Server
gnome-terminal --title="Server" -- bash -c "
cd $WORK_DIR
source $VENV_ACTIVATE
export PYTHONPATH=$PROJECT_ROOT
python federated_main.py --mode server --config federated_config.yaml
exec bash
"

sleep 20

# Terminal 2: SST2 Client
gnome-terminal --title="SST2 Client" -- bash -c "
cd $WORK_DIR
source $VENV_ACTIVATE
export PYTHONPATH=$PROJECT_ROOT
python federated_main.py --mode client --client_id sst2_client --tasks sst2
exec bash
"

# Terminal 3: QQP Client
gnome-terminal --title="QQP Client" -- bash -c "
cd $WORK_DIR
source $VENV_ACTIVATE
export PYTHONPATH=$PROJECT_ROOT
python federated_main.py --mode client --client_id qqp_client --tasks qqp
exec bash
"

# Terminal 4: STSB Client
gnome-terminal --title="STSB Client" -- bash -c "
cd $WORK_DIR
source $VENV_ACTIVATE
export PYTHONPATH=$PROJECT_ROOT
python federated_main.py --mode client --client_id stsb_client --tasks stsb
exec bash
"

echo "All terminals launched!"
