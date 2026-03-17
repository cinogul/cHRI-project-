#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import socket

# UDP
s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s_in.bind(("127.0.0.1", 5005))
s_in.setblocking(False)

s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_out.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# dummy send
s_out.sendto(np.zeros(2).tobytes(), ("127.0.0.1", 5006))

# flush UDP
while True:
    try:
        s_in.recvfrom(1024)
    except:
        break

# received values
ext_pct = 0.0
angle   = 0.0
Kx      = 1000.0
Ky      = 100.0

# MAIN LOOP
run = True
while run:
    # UDP In
    try:
        data, addr = s_in.recvfrom(1024)
        packet = np.frombuffer(data, dtype=np.float64)
        if len(packet) >= 5:
            ext_pr = packet[0]
            angle = packet[1]
            Kx = packet[2]
            Ky = packet[3]
            cam = packet[4]
            height = packet[5]
    except:
        pass

    # UDP Out - F
    F = np.zeros(2)
    s_out.sendto(F.tobytes(), ("127.0.0.1", 5006))

s_in.close()
s_out.close()