#!/bin/bash
#SBATCH --nodes=2
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --constraint=cpu
#SBATCH --job-name=ogbn-arxiv
#SBATCH --output=ogbn-arxiv-%j.out
#SBATCH --error=ogbn-arxiv-%j.err

ulimit -c unlimited
ulimit -v unlimited
source activate dgl
PYTHON_PATH=$(which python)

DATASET_NAME="ogbn-arxiv"
PARTITION_METHOD="metis"
NUM_NODES=$SLURM_JOB_NUM_NODES
SAMPLER_PROCESSES=0
BACKEND="gloo"
TRAINERS=4
JOBID=$SLURM_JOB_ID

PROJ_PATH=$(pwd)

# these data dirs are important to run the script
DATA_DIR="/pscratch/sd/s/sark777/Distributed_DGL/dataset"
PARTITION_DIR="/pscratch/sd/s/sark777/Distributed_DGL/partitions/${PARTITION_METHOD}/${DATASET_NAME}/${NUM_NODES}_parts/${DATASET_NAME}.json"
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST) # get list of nodes


# append job id to IP_CONFIG_FILE
IP_CONFIG_FILE="ip_config_${SLURM_JOB_ID}.txt"


echo "Generating ip_config.txt..."
: > $IP_CONFIG_FILE  # Empty the file if it exists

for node in $NODELIST; do
    echo "Getting first 10.249.x.x IP address for node: $node"

    # Use srun to execute 'ip addr show' on each node and extract the first 10.249.x.x IP
    first_ip=$(srun --nodes=1 --nodelist=$node ip addr show | grep 'inet 10.249' | awk '{print $2}' | cut -d'/' -f1 | head -n 1)

    # Check if an IP address was found and append it to the ipconfig file
    if [ ! -z "$first_ip" ]; then
        echo $first_ip >> $IP_CONFIG_FILE
    else
        echo "No 10.249.x.x IP found for $node" >&2
    fi
done

# Print the contents of the ipconfig file
cat $IP_CONFIG_FILE

# assert that the number of lines in ip_config.txt is equal to NUM_NODES
NUM_IPS=$(wc -l < $IP_CONFIG_FILE)
if [ "$NUM_IPS" -ne "$NUM_NODES" ]; then
    echo "Number of IPs ($NUM_IPS) does not match number of nodes ($NUM_NODES)"
    exit 1
fi


# Run the training script
$PYTHON_PATH $PROJ_PATH/launch.py \
--workspace $PROJ_PATH \
--num_trainers $TRAINERS \
--num_samplers $SAMPLER_PROCESSES \
--num_servers 1 \
--part_config $PARTITION_DIR \
--ip_config  $IP_CONFIG_FILE \
"$PYTHON_PATH main.py --graph_name $DATASET_NAME \
--backend $BACKEND \
--ip_config $IP_CONFIG_FILE"
