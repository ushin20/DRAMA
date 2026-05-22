import os
import numpy as np
import torch
import math
from collections import deque
from typing import List, Dict
import re

def parse_device_info(device_ids: List[str]) -> Dict[str, Dict[str, str]]:
    sensor_prefixes = ['LIT', 'FIT', 'AIT', 'DPIT', 'PIT']
    actuator_prefixes = ['P', 'MV', 'UV']
    result = {}
    
    for device in device_ids:
        digits = ''.join(filter(str.isdigit, device))
        process_num = f"P{digits[0]}" if digits else "unknown"
        
        prefix_match = re.match(r'[A-Z]+', device)
        prefix = prefix_match.group(0) if prefix_match else "unknown"
        
        if prefix in sensor_prefixes:
            device_role = "sensor"
        elif prefix in actuator_prefixes:
            device_role = "actuator"
        else:
            device_role = "unknown"
        
        result[device] = {
            "id": process_num,
            "role": device_role,
            "type": prefix
        }
    
    return result

def parse_device_info_WADI(device_ids):
    sensor_prefixes = ['FIT', 'AIT', 'DPIT', 'PIT', 'LT', 'LS', 'FIC', 'PIC', 'FQ']
    actuator_prefixes = ['P', 'MV', 'SV', 'MCV']
    
    result = {}
    for device in device_ids:
        parts = device.split('_')
        
        # process ID
        process_id = parts[0]  # e.g., '1', '2', '2A', '3'
        if process_id in ['1', '3']:
            pass
        elif process_id == '2':
            # -- P2A
            if parts[1] in ['MV'] and parts[2] in ['001','002','003','004']:
                process_id = 'P2A'
            elif parts[1] in ['LT', 'LS'] and parts[2] in ['001', '002']:
                process_id = 'P2A'
            elif parts[1] in ['PIT', 'FIT'] and parts[2] in ['001']:
                process_id = 'P2A'

            # -- P2B
            elif parts[1] == 'MV' and parts[2] in ['005', '006', '009']:
                process_id = 'P2B'
            elif parts[1] in ['PIT', 'FIT'] and parts[2] in ['002', '003']:
                process_id = 'P2B'
            elif parts[1] == 'P' and parts[2] in ['003', '004']:
                process_id = 'P2B'
            elif parts[1] == 'DPIT':
                process_id = 'P2B'
            elif parts[1] == 'MCV' and parts[2] == '007':
                process_id = 'P2B'
            elif parts[1] == 'PIC' and parts[2] == '003':
                process_id = 'P2B'
            

            # -- P2C
            elif parts[2] == '101':
                process_id = 'P2C'
            elif parts[2] == '201':
                process_id = 'P2C'
            elif parts[2] == '301':
                process_id = 'P2C'
            elif parts[2] == '401':
                process_id = 'P2C'
            elif parts[2] == '501':
                process_id = 'P2C'
            elif parts[2] == '601':
                process_id = 'P2C'

            # -- else
            else:
                raise NotImplementedError(f'Check {device}...')
        elif process_id == '2A':
            process_id = 'P2A'
        elif process_id == '2B':
            process_id = 'P2B'
        else:
            process_id = 'global'

        prefix = parts[1] if len(parts) > 1 else "unknown"
        suffix = parts[-1] if len(parts) > 1 else "unknown"
        sensor_exceptions = {'2_P_003_SPEED', '2_P_004_SPEED'}
        actuator_exceptions = {'2_PIC_003_SP'}
        if device in sensor_exceptions:
            role = 'sensor'
        elif device in actuator_exceptions:
            role = 'actuator'
        # General rule
        elif suffix in ['STATUS', 'AH', 'AL']:
            role = 'actuator'
        elif suffix in ['PV', 'SP', 'CO']:
            role = 'sensor'
        elif prefix in actuator_prefixes:
            role = 'actuator'
        elif prefix in sensor_prefixes:
            role = 'sensor'
        else:
            role = 'unknown'
        
        # prefix edit
        if process_id == 'global':
            prefix = device
        result[device] = {
            "id": process_id,
            "role": role,
            "type": prefix
        }

        #print(device, result[device])

    return result

def build_spatial_edge(mode='swat',verbose=False):
    if mode == 'smap':
        C = 25
    elif mode == 'msl':
        C = 55
    else:
        with open(os.path.join('data', mode, 'preprocessed', 'columns.txt'), 'r') as f:
            sensor_list = [line.strip() for line in f.readlines()]
            print(f"Number of sensors in {mode}: {len(sensor_list)}")
            if verbose:
                print(f"Sensor list of {mode}:")
                print(*sensor_list, sep=', ')

        C = len(sensor_list)

    edge = torch.eye(C)

    if mode=='swat':
        device_info = parse_device_info(sensor_list)

        process = {
            'P1':'P2',
            'P2':'P3',
            'P3':'P4',
            'P4':'P5',
            'P5':'P6',
            'P6':'P3',
        }
        inverse_process = {v: k for k, v in process.items()}
        for i,a in enumerate(sensor_list):
            a_info = device_info[a]
            for j,b in enumerate(sensor_list):
                if a == b:
                    continue
                
                b_info = device_info[b]

                # Intra-process control: S→A or A→S
                if a_info['id'] == b_info['id']:
                    if (a_info['role'], b_info['role']) in [('sensor', 'actuator'), ('actuator', 'sensor')]:
                        edge[i][j] = 1

                # Intra-process similarity: S↔S or A↔A
                if a_info['id'] == b_info['id'] and a_info['role'] == b_info['role'] and a_info['type'] == b_info['type']:
                    edge[i][j] = 1

                # Process flow (P_i → P_j)
                if process.get(a_info['id'], None) == b_info['id']:
                    edge[j][i] = 1

                # Feedback edge: sensor in P_i → actuator in P_j, wherer j = i-1
                if a_info['role'] == 'sensor' and b_info['role'] == 'actuator':
                    if inverse_process.get(a_info['id'], None) == b_info['id']:
                        edge[j][i] = 1

    elif mode=='wadi':
        device_info = parse_device_info_WADI(sensor_list)
        process = { 
            '1':'P2A',
            'P2A': 'P2B',
            'P2B': 'P2C',
            'P2C': '3',
        }

        from collections import defaultdict
        inverse_process = defaultdict(list)
        for src, dst in process.items():
            if isinstance(dst, list):
                for d in dst:
                    inverse_process[d].append(src)
            else:
                inverse_process[dst].append(src)

        for i, a in enumerate(sensor_list):
            a_info = device_info[a]
            a_pid = a_info['id'] # 1,

            for j, b in enumerate(sensor_list):
                if a == b:
                    continue
                b_info = device_info[b]
                b_pid = b_info['id']

                # (1) Intra-process 
                if a_pid == b_pid:
                    if (a_info['role'], b_info['role']) in [('sensor', 'actuator'), ('actuator', 'sensor')]:
                        edge[i][j] = 1
                        edge[j][i] = 1
                    if a_info['role'] == b_info['role'] and a_info['type'] == b_info['type']:
                        edge[i][j] = 1
                        edge[j][i] = 1

                # (2) Process flow: a ∈ P_k, b ∈ P_{k+1}
                next_pids = process.get(a_pid, [])
                if not isinstance(next_pids, list):
                    next_pids = [next_pids]

                if b_pid in next_pids:
                    # P_a -> P_b
                    # all[i] => all[j]
                    edge[j][i] = 1

                # (3) Inverse process flow: b ∈ P_{k-1}, a ∈ P_k
                prev_pids = inverse_process.get(a_pid, [])
                if b_pid in prev_pids and (b_info['role'], a_info['role']) in [('sensor', 'actuator')]:
                    # P_b <- P_a
                    # sensor[j] ← actuator[i]
                    edge[j][i] = 1 
                
    elif mode == 'msl' or mode == 'smap':
        telemetry_idx = 0

        for i in range(1, C - 1, 2):  # i = 1, 3, 5, ...
            edge[i][i + 1] = 1              # cmd → cmd_received
            edge[i + 1][telemetry_idx] = 1  # cmd_received → telemetry

    return edge


def check_connection(edge: torch.Tensor, sensor_names: List[str], src: str, dst: str):
    i = sensor_names.index(src)
    j = sensor_names.index(dst)
    forward = bool(edge[j, i])
    backward = bool(edge[i, j])
    return {
        f"{src} → {dst}": forward,
        f"{dst} → {src}": backward
    }


if __name__ == '__main__':
    target = 'wadi'
    edge = build_spatial_edge(target)
    with open(os.path.join('data', target, 'preprocessed', 'columns.txt'), 'r') as f:
        sensor_names = [line.strip() for line in f.readlines()]
    C = edge.shape[0]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 12))
    plt.imshow(edge, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Edge')
    plt.title(f"{target.upper()} adjacency matrix")
    plt.xticks(ticks=range(C), labels=sensor_names, rotation=90, fontsize=4)
    plt.yticks(ticks=range(C), labels=sensor_names, fontsize=6)
    plt.xlabel("To Sensor")
    plt.ylabel("From Sensor")
    plt.tight_layout()
    plt.show()
    plt.close()