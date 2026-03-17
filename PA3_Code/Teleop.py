# -*- coding: utf-8 -*-
import sys
import math
import time
import numpy as np
import pygame

from Physics import Physics
from GraphicsModified import Graphics

import socket

class PA:
    def __init__(self):
        self.physics = Physics(hardware_version=2)
        self.device_connected = self.physics.is_device_connected()
        self.graphics = Graphics(self.device_connected)
        self.graphics.effort_color = (255, 255, 255)
        xc,yc = self.graphics.screenHaptics.get_rect().center

        ##############################################
        # UDP
        self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # extension%, angle, stiffness to environment.py
        self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s_in.bind(("127.0.0.1", 5006))  # F from environment.py
        self.s_in.setblocking(False)

        self.F_feedback = np.zeros(2)

        # stiffness state (tele-impedance)
        self.Ks = np.diag([1000.0, 100.0])
        self.stiffness_value_increment = 100.0
        
        # camera view
        self.cam = 0
        
        # crane height
        self.height = 0
        self.height_increment  = 0.05

        self.font = pygame.font.Font('freesansbold.ttf', 12)
        self.graphics.show_debug = False

        # dummy send
        self.s_out.sendto(np.zeros(4).tobytes(), ("127.0.0.1", 5005))
        
        ##############################################

    def run(self):
        p = self.physics
        g = self.graphics
        keyups,xm = g.get_events()

        if self.device_connected:
            pA0,pB0,pA,pB,pE = p.get_device_pos()
            pA0,pB0,pA,pB,xh = g.convert_pos(pA0,pB0,pA,pB,pE)
        else:
            xh = g.haptic.center

        fe = np.array([0.0,0.0])
        xh = np.array(xh)
        xc,yc = g.screenHaptics.get_rect().center
        g.erase_screen()

        ##############################################
        for key in keyups:
            if key==ord("q"):
                sys.exit(0)
            if key == ord('m'):
                pygame.mouse.set_visible(not pygame.mouse.get_visible())
            if key == ord('r'):
                g.show_linkages = not g.show_linkages

            # tele-impedance stiffness control
            if key == pygame.K_RIGHT:
                self.Ks[0,0] = min(1000.0, self.Ks[0,0] + self.stiffness_value_increment)
            if key == pygame.K_LEFT:
                self.Ks[0,0] = max(0.0, self.Ks[0,0] - self.stiffness_value_increment)
            if key == pygame.K_UP:
                self.Ks[1,1] = min(1000.0, self.Ks[1,1] + self.stiffness_value_increment)
            if key == pygame.K_DOWN:
                self.Ks[1,1] = max(0.0, self.Ks[1,1] - self.stiffness_value_increment)
            if key == pygame.K_KP1:
                self.cam = 0
            if key == pygame.K_KP2:
                self.cam = 1
            if key == pygame.K_KP3:
                self.cam = 2
            if key == pygame.K_KP5:
                self.height = min(3.2, self.height + self.height_increment)
            if key == pygame.K_KP8:
                self.height = max(0.0, self.height - self.height_increment)
                
            
            

        # haptic pos -> robot base frame
        pr = np.array([(xh[0] - xc) / 800,
                       -(xh[1] - yc) / 800])

        # physical position and derived metrics
        pos_phys = g.inv_convert_pos(xh)
        min_ext = p.l2 - p.l1
        max_ext = 0.151 # hardcoded max haply extension
        ext_pct = (np.linalg.norm(pos_phys) - min_ext) / (max_ext - min_ext) * 100.0
        ext_pct = np.clip(ext_pct, 0.0, 100.0)
        angle = math.atan2(pos_phys[0], pos_phys[1])  # angle from vertical axis through device base

        # UDP Out - extension %, angle, Ks
        packet = np.array([ext_pct, angle, self.Ks[0,0], self.Ks[1,1], self.cam])
        self.s_out.sendto(packet.tobytes(), ("127.0.0.1", 5005))

        # UDP In - F
        try:
            data, addr = self.s_in.recvfrom(1024)
            self.F_feedback = np.frombuffer(data, dtype=np.float64)
        except:
            pass

        # scaled force feedback
        force_feedback_scale = -0.05
        fe += self.F_feedback * force_feedback_scale

        # stiffness legend
        kx_surf = self.font.render("Kx (Left/Right) = {:.0f}".format(self.Ks[0,0]), True, (0, 0, 0), (255, 255, 255))
        ky_surf = self.font.render("Ky (Up/Down) = {:.0f}".format(self.Ks[1,1]), True, (0, 0, 0), (255, 255, 255))
        cam_surf = self.font.render("Camera View (1/2/3) = {:.0f}".format(self.cam+1), True, (0, 0, 0), (255, 255, 255))
        height_surf = self.font.render("Height (8/5) = {:.2f}".format(self.height), True, (0, 0, 0), (255, 255, 255))

        debug_surf = self.font.render("x={:.3f}  y={:.3f}  dist={:.3f}m  ext={:.1f}%  angle={:.2f}rad".format(
            pr[0], pr[1], np.linalg.norm(pos_phys), ext_pct, angle), True, (0, 0, 0), (255, 255, 255))
        g.screenHaptics.blit(kx_surf, (10, 10))
        g.screenHaptics.blit(ky_surf, (10, 25))
        g.screenHaptics.blit(cam_surf, (10,40))
        g.screenHaptics.blit(height_surf, (10, 55))
        g.screenHaptics.blit(debug_surf, (10, 70))

        ##############################################
        if self.device_connected:
            fe_device = fe * np.array([-1, -1])
            p.update_force(fe_device)
        else:
            xh = g.sim_forces(xh, -fe, xm, mouse_k=0.5, mouse_b=0.8)
            pos_phys = g.inv_convert_pos(xh)
            pA0,pB0,pA,pB,pE = p.derive_device_pos(pos_phys)
            pA0,pB0,pA,pB,xh = g.convert_pos(pA0,pB0,pA,pB,pE)
        g.render(pA0,pB0,pA,pB,xh,fe,xm)

    def close(self):
        ##############################################
        self.s_in.close()
        self.s_out.close()
        ##############################################
        self.graphics.close()
        self.physics.close()

if __name__=="__main__":
    pa = PA()
    try:
        while True:
            pa.run()
    finally:
        pa.close()