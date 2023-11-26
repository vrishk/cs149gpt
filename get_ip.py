#!/usr/bin/env python3

# This script will check the available machines in the shared cluster and
# return one of the IP addresses.

# All native python libraries, so no need to install anything.
import argparse
import random
import re
from urllib.request import urlopen
from multiprocessing import Pool


CLUSTER_URL = 'https://web.stanford.edu/~tomshen/cs149/server_list.txt'

STATUS_SERVER_PORT = 17777


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status-server-port', default=STATUS_SERVER_PORT)
    return parser.parse_args()


def check_liveness(ip):
    try:
        r = urlopen('http://{}:{}/status'.format(ip, STATUS_SERVER_PORT)
                    , timeout=3).read().decode('ascii')
        # 23:34:05 up 3 days, 23:35, 5 users, load average: 2.65, 3.00, 3.30
        m = re.match(r'.*(\d+) user', r)
    except Exception as e:
        return False, 0xFFFFFFFF
    if not m:
        return False, 0xFFFFFFFF
    return True, int(m.group(1))


def main(args):
    global STATUS_SERVER_PORT
    STATUS_SERVER_PORT = args.status_server_port

    r = urlopen(CLUSTER_URL).read().decode('ascii')

    all_ips = []
    for ip in r.split('\n'):
        ip = ip.strip()
        if ip:
            all_ips.append(ip)
    random.shuffle(all_ips)

    with Pool() as p:
        status = p.map(check_liveness, all_ips)
    all_ips = [(x, status[i][1]) for i, x in enumerate(all_ips) if status[i][0]]
    all_ips.sort(key=lambda x: x[1])

    print('Cluster IP addresses:')
    if len(all_ips) > 0:
        min_load = min(x[1] for x in all_ips)
        for ip, load in all_ips:
            if load == min_load:
                print(' {} <--- least loaded'.format(ip))
            else:
                print(' {}'.format(ip))
    else:
        print('None available... Check that you are connected to the Internet.')


if __name__ == '__main__':
    main(get_args())