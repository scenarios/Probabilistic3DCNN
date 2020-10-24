sudo apt-get update
sudo apt-get install jq -y

#this_container_name=`cat /proc/sys/kernel/hostname`
#this_container_index=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$this_container_name" -r '.containers[$name].index'`

#cp $PHILLY_RUNTIME_CONFIG /log/envrecord

all_containers=`cat $PHILLY_RUNTIME_CONFIG | jq ".containers[].id"`
for container in $all_containers
do
        #echo $container
        container=${container//\"/}
        #echo "cat $PHILLY_RUNTIME_CONFIG | jq --arg name $container -r '.containers[\"container_e125_1515549374026_2107_01_000002\"].index'"
        index=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].index'`
        if [ $index -eq 0 ]
        then
                master_ip=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].ip'`
                master_port_start=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].portRangeStart'`
                master_port_end=`cat $PHILLY_RUNTIME_CONFIG | jq --arg name "$container" -r '.containers[$name].portRangeEnd'`
                DIFF=$((master_port_end-master_port_start+1))
        fi
done

this_container_index=$PHILLY_CONTAINER_INDEX

export NODE_RANK=$this_container_index
export MASTER_IP=$master_ip
#export MASTER_PORT=$(($(($RANDOM%$DIFF))+master_port_start))
export MASTER_PORT=$((master_port_start+1))

echo $MASTER_IP
echo $MASTER_PORT

echo `ls /`