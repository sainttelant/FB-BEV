#build the dockerfile

docker build -t fb_occ:v2 -f Dockerfile .

# run images container
docker run --privileged -it --gpus '"device=0"' --shm-size=16g -v /home/wilsxue/Bev:/workspace/Bev -v /home/Cnworkspace/nuscenes:/workspace/nuscenes --name FB_OCC_official fb_occ:v2  /bin/bash

# for evaluation
