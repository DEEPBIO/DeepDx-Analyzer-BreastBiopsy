#!/bin/bash
# Run tests
num_cpus=$(grep -c ^processor /proc/cpuinfo)
docker_cpus=$(($num_cpus>16?16:num_cpus))

echo Num of cores used: $docker_cpus

# TODO: some version of docker and nvidia-docker seems to need gpus option, but some does not. SHOULD CHECK
docker run -i --rm \
--gpus=all \
--cpus=$docker_cpus \
--shm-size=64g \
-e PYTHON_ENV=test \
-e NVIDIA_VISIBLE_DEVICES=0 \
-v $(pwd):/root/code \
-v /mnt/nfs0:/mnt/nfs0 \
-v /mnt/devstorage/:/unittest_data_storage \
deepdx-analyzer-breast-unittest \
bash -c "cd /root/code && python3 -u -m unittest -f tests/spec_1_input_test.py tests/spec_2_unit_test.py tests/spec_3_invariance_test.py"
