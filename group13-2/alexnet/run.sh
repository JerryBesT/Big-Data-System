#!/bin/bash

source cluster_utils.sh
terminate_cluster
start_cluster startserver.py cluster2
python -m AlexNet.scripts.train --mode cluster2

source cluster_utils.sh
terminate_cluster
start_cluster startserver.py cluster
python -m AlexNet.scripts.train --mode cluster
