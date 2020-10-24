import os
import time


if __name__ == '__main__':
    with open(os.path.join(os.environ['HPC_RUNTIMECONFIG_PATH'], os.environ['THIS_HOSTNAME']+'.txt'), 'w') as f:
        f.write('PLACEHOLDER')
    time.sleep(15)
    while len([host for host in os.listdir(os.environ['HPC_RUNTIMECONFIG_PATH'])]) < int(os.environ['NNODES']):
        time.sleep(5)

    host_list = [host.strip('.txt') for host in os.listdir(os.environ['HPC_RUNTIMECONFIG_PATH'])]
    assert len(host_list) == int(os.environ['NNODES']), 'number of host must be equal to number of nodes ' + str(len(host_list)) + ' vs ' + str(int(os.environ['NNODES']))
    host_list.sort()
    host_dict = dict(zip(host_list, range(0, len(host_list))))

    this_container_index = host_dict[os.environ['THIS_HOSTNAME']]
    master_ip = host_list[0]
    master_port = 6000

    print('export NODE_RANK={}'.format(this_container_index))
    print('export MASTER_IP={}'.format(master_ip))
    print('export MASTER_PORT={}'.format(master_port))