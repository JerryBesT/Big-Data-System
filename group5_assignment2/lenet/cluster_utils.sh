#!/bin/bash
export TF_RUN_DIR="~/tf"

function terminate_cluster() {
    echo "Terminating the servers"
    CMD="ps aux | grep -v 'grep' | grep -v 'bash' | grep -v 'ssh' | grep 'python lenet.py' | awk -F' ' '{print \$2}' | xargs kill -9"
    for i in `seq 0 2`; do
        ssh node$i "$CMD"
    done
}

function start_cluster() {
    if [ -z $2 ]; then
        echo "Usage: start_cluster <python script> <number of workers>"
        echo "Here, <python script> contains the cluster spec that assigns an ID to all server."
    else
        echo "Create $TF_RUN_DIR on remote hosts if they do not exist."
        echo "Copying the script to all the remote hosts."
        for i in `seq 0 2`; do
            ssh node$i "mkdir -p $TF_RUN_DIR"
            scp $1 node$i:$TF_RUN_DIR
        done
        if [ "$2" = "1" ]; then
            nohup ssh node0 "cd ~/tf ; python $1 --num_worker=1  --task_index=0" > serverlog-1-0.out 2>&1&
        elif [ "$2" = "2" ]; then
            nohup ssh node0 "cd ~/tf ; python $1 --num_worker=2  --task_index=0" > serverlog-2-0.out 2>&1&
            nohup ssh node1 "cd ~/tf ; python $1 --num_worker=2  --task_index=1" > serverlog-2-1.out 2>&1&
        else
            nohup ssh node0 "cd ~/tf ; python $1 --num_worker=3  --task_index=0" > serverlog-3-0.out 2>&1&
            nohup ssh node1 "cd ~/tf ; python $1 --num_worker=3  --task_index=1" > serverlog-3-1.out 2>&1&
            nohup ssh node2 "cd ~/tf ; python $1 --num_worker=3  --task_index=2" > serverlog-3-2.out 2>&1&
        fi
    fi
}