#!/bin/bash

source cluster_utils.sh

terminate_cluster

start_cluster startserver.py cluster
