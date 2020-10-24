#source /etc/os-release
#apt-get update
#apt-get install wget -y
#wget https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb
#dpkg -i packages-microsoft-prod.deb
#apt-get update && apt-get install libcurl3-gnutls $(apt-cache search libgnutls[0-9][0-9]|cut -d' ' -f 1)  libfuse2 blobfuse -y
#rm packages-microsoft-prod.deb

#export AZURE_STORAGE_ACCOUNT=yizhouautoml
#export AZURE_STORAGE_ACCESS_KEY=lBBOIqJx1dffBxaTCIEJ0JRdS5FjfvdoGllPF02toRrtXR9YmWOrK8ULuhqWX93iAATVV7rFOHhITyLPCIooxg==

#mkdir /mnt/blobfusetmp/
#blobfuse /code --container-name=code --tmp-path=/mnt/blobfusetmp
#mkdir /log
#mkdir /data
#mkdir /gcrhpc-runtime-sharefile
#mkdir /mnt/bfs_log_tmp/
#mkdir /mnt/bfs_data_tmp/
#mkdir /mnt/bfs_share_tmp/

#blobfuse /log --container-name=log --tmp-path=/mnt/blob_log_tmp
#blobfuse /data --container-name=data --tmp-path=/mnt/blob_data_tmp
#blobfuse /gcrhpc-runtime-sharefile --container-name=gcrhpc-runtime-sharefile #--tmp-path=/mnt/blob_share_tmp

#export HPC_RUNTIMECONFIG_PATH="/gcrhpc-runtime-sharefile"
#export THIS_HOSTNAME=`hostname`

#cd /code/NAS_spatiotemporal
#eval `python setup_hpc.py`

#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_IP --master_port=$MASTER_PORT main.py --node_rank=$NODE_RANK --batch-size=10 --dropout=0.2 --epochs=50 --lr_steps 20 30 40 --num_dense_sample=32 --dense_sample_stride=1 --dataset=something --root_log=/log/log --root_model=/log/checkpoint --root_path=/data --print-freq=50 --eval-freq=2
#python -m torch.distributed.launch --nproc_per_node=4 main.py --node_rank=0 --batch-size=10 --dropout=0.9 --epochs=50 --lr_steps 20 30 40 --num_dense_sample=32 --dense_sample_stride=1 --dataset=something --root_log=/log/log --root_model=/log/checkpoint --root_path=/data --print-freq=50 --eval-freq=2
if [ "$4" == "True" ]
then
    if [ "$6" == "True" ]
    then
        python -m torch.distributed.launch --nproc_per_node=16 main.py --suffix=Net"$1"Bz"$5"by16Lr"$2"SbnTpbtShareStp40-60-80 --enable_nasas --node_rank=0 --net_version="$1" --syncbn --temporal_nasas_only --lr="$2" --batch-size="$5" --dropout=0.0 --weight_decay=0.0 --weight_reg="$3" --epochs=90 --lr_steps 40 60 80 --dense_sample --num_dense_sample=32 --dense_sample_stride=1 --dataset="$7" --training_size="$8" --root_log=/msraimscratch/v-yizzh/log/log --root_model=/msraimscratch/v-yizzh/log/checkpoint --root_path=/raid/yizzh/data --resume=/msraimscratch/v-yizzh/log/checkpoint/"$9"/ckpt.best.pth.tar --print-freq=200 --eval-freq=4 --workers=8 --warmup --philly-nccl-multi-node --break_resume --use_zip
    else
        python -m torch.distributed.launch --nproc_per_node=16 main.py --suffix=Net"$1"Bz$5$by16Lr"$2"SbnStp40-60-80 --enable_nasas --node_rank=0 --net_version="$1" --syncbn --lr="$2" --batch-size="$5" --dropout=0.0 --weight_decay=0.0 --weight_reg="$3" --epochs=90 --lr_steps 40 60 80 --dense_sample --num_dense_sample=32 --dense_sample_stride=1 --dataset="$7" --training_size="$8" --root_log=/msraimscratch/v-yizzh/log/log --root_model=/msraimscratch/v-yizzh/log/checkpoint --root_path=/raid/yizzh/data --resume=/msraimscratch/v-yizzh/log/checkpoint/"$9"/ckpt.best.pth.tar --print-freq=200 --eval-freq=4 --workers=8 --warmup --philly-nccl-multi-node --break_resume --use_zip
    fi
else
    if [ "$6" == "True" ]
    then
        python -m torch.distributed.launch --nproc_per_node=16 main.py --suffix=Net"$1"Bz"$5"by16Lr"$2"NsbnTpbtShareStp40-60-80 --enable_nasas --node_rank=0 --net_version="$1" --temporal_nasas_only --lr="$2" --batch-size="$5" --dropout=0.0 --weight_decay=0.0 --weight_reg="$3" --epochs=90 --lr_steps 40 60 80 --dense_sample --num_dense_sample=32 --dense_sample_stride=1 --dataset="$7" --training_size="$8" --root_log=/msraimscratch/v-yizzh/log/log --root_model=/msraimscratch/v-yizzh/log/checkpoint --root_path=/raid/yizzh/data --resume=/msraimscratch/v-yizzh/log/checkpoint/"$9"/ckpt.best.pth.tar --print-freq=200 --eval-freq=4 --workers=8 --warmup --philly-nccl-multi-node --break_resume --use_zip
    else
        python -m torch.distributed.launch --nproc_per_node=16 main.py --suffix=Net"$1"Bz$5$by16Lr"$2"NsbnStp40-60-80 --enable_nasas --node_rank=0 --net_version="$1" --lr="$2" --batch-size="$5" --dropout=0.0 --weight_decay=0.0 --weight_reg="$3" --epochs=90 --lr_steps 40 60 80 --dense_sample --num_dense_sample=32 --dense_sample_stride=1 --dataset="$7" --training_size="$8" --root_log=/msraimscratch/v-yizzh/log/log --root_model=/msraimscratch/v-yizzh/log/checkpoint --root_path=/raid/yizzh/data --resume=/msraimscratch/v-yizzh/log/checkpoint/"$9"/ckpt.best.pth.tar --print-freq=200 --eval-freq=4 --workers=8 --warmup --philly-nccl-multi-node --break_resume --use_zip
    fi
fi