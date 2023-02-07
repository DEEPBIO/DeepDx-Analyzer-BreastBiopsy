#!/bin/bash
# Update test snapshots
docker run -i --rm \
--gpus=all \
--shm-size=64g \
-e PYTHON_ENV=test \
-e NVIDIA_VISIBLE_DEVICES=0 \
-v $(pwd):/root/code \
-v /mnt/nfs0:/mnt/nfs0 \
-v /mnt/devstorage/:/unittest_data_storage \
deepdx-analyzer-breast-unittest \
bash -c "cd /root/code && python3 tests/update_snapshots.py"
