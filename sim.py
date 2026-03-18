import mujoco
import mujoco.viewer
import math
import time
import numpy as np
import socket

# UDP
s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s_in.bind(("127.0.0.1", 50005))
s_in.setblocking(False)

s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_out.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

s_out2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_out2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# dummy send
s_out.sendto(np.zeros(2).tobytes(), ("127.0.0.1", 50006))
s_out2.sendto(np.zeros(2).tobytes(), ("127.0.0.1", 50007))

# flush UDP
while True:
    try:
        s_in.recvfrom(1024)
    except:
        break

# received values
ext_pct     = 0.0
angle       = 0.0
ext_enabled = True
rot_enabled = True
cam         = 0
height      = 0
force_world = np.zeros(3)

# wind assistance parameters
C       = 9
scaling = 1 / 35

def get_assistance_force(wind_vec):
    wx, wy = wind_vec[0], wind_vec[1]
    wind_speed = math.sqrt(wx**2 + wy**2)
    if wind_speed == 0:
        return np.zeros(2)
    wind_dir  = math.degrees(math.atan2(wy, wx))
    magnitude = C * (wind_speed ** 2)
    F_ax = -magnitude * math.cos(math.radians(wind_dir)) * scaling
    F_ay = -magnitude * math.sin(math.radians(wind_dir)) * scaling
    if math.sqrt(F_ax**2 + F_ay**2) > 4:
        return np.array([F_ax, F_ay])
    return np.zeros(2)

xml = """
<mujoco>
  <option wind="0 0 0" density="1.2" viscosity="0.000018"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    
    <mesh name="crane_mesh" file="crane.stl"/>
    <mesh name="turbine_base" file="turbine_base.stl"/>
    <mesh name="turbine_block" file="turbine_block.stl"/>
  </asset>

  <worldbody>
    <light pos="0 0 10" dir="0 0 -1" directional="true"/>
    
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <camera name="global_overview" pos="-5 -10 20" mode="targetbodycom" target="crane_gantry"/>

    <body name="turbine_base_body" pos="4 10 0">
      <inertial pos="0 0 0" mass="20000" diaginertia="10000 10000 15000"/>
      <geom type="mesh" mesh="turbine_base" contype="1" conaffinity="1" rgba="1 1 1 1"/>
    </body>

    <body name="crane" pos="0 0 0">
      <inertial pos="0 0 0" mass="50000" diaginertia="20000 20000 30000"/>
      <joint name="crane_base_hinge" type="hinge" axis="0 0 -1" damping="1000000" range="-90 90" limited="true"/>
      <camera name="crane_low_view" pos="0.5 1 10" mode="targetbody" target="payload"/>
      <geom type="cylinder" size="0.5 0.5" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="crane_mesh" pos="0.3 2.2 0" contype="0" conaffinity="0"/>

      <body name="crane_gantry" pos="0 2 8.2">
        <inertial pos="0 0 0" mass="5000" diaginertia="500 500 500"/>
        <joint name="trolley_slide" type="slide" axis="0 1 0" range="0.0 5.0" limited="true" damping="5000"/>
        <geom type="box" size="0.15 0.15 0.05" contype="0" conaffinity="0"/>

        <body name="payload" pos="0 0 -5">
          <inertial pos="0 0 2.5" mass="100" diaginertia="20 20 2"/>
          <joint name="cable_swing" type="ball" pos="0 0 5" damping="1.0"/>
          <geom type="capsule" size="0.02 2.5" pos="0 0 2.5" contype="0" conaffinity="0"/>
          <body name="turbine_block_body" pos="0 0 0">
            <joint name="cable_slide" type="slide" axis="0 0 -1" range="0.0 3.2" limited="true"/>
            <inertial pos="0 0 0" mass="10000" diaginertia="3000 3000 3000"/>
            <camera name="block_cam" pos="0 0 1" euler="-1.57 0 0" fovy="60"/>
            <geom type="mesh" mesh="turbine_block" contype="1" conaffinity="1" rgba="1 1 1 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
  <position name="cable_extension" joint="cable_slide" 
            kp="2000000" kv="200000" ctrlrange="0 3.2" forcerange="-10000000000 10000000000"/>
</actuator>


</mujoco>
"""

model   = mujoco.MjModel.from_xml_string(xml)
data    = mujoco.MjData(model)
body_name = "turbine_block_body"
body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
base_name = "turbine_base_body"
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_name)

j = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.fixedcamid = 0
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    while viewer.is_running():
        j += 1
        step_start = time.time()

        try:
            info, addr = s_in.recvfrom(1024)
            packet = np.frombuffer(info, dtype=np.float64)
            if len(packet) >= 6:
                ext_pct     = packet[0]
                angle       = packet[1]
                ext_enabled = bool(packet[2])
                rot_enabled = bool(packet[3])
                cam         = int(packet[4])
                height      = packet[5]
        except:
            pass

        # crane angle
        if rot_enabled:
            data.qpos[0] = angle

        # trolley depth
        if ext_enabled:
            data.qpos[1] = ext_pct * 5 / 100

        # block height
        data.ctrl[0] = height

        viewer.cam.fixedcamid = cam

        model.opt.wind = [30 * np.sin(j / 1000), 0, 0]

        # contact force
        F = np.zeros(2)        
        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2
            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]
            if b1 == body_id or b2 == body_id:
                cf = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, i, cf)
                frame = contact.frame.reshape(3, 3)
                force_world = frame.T @ cf[:3]
                F = force_world[:2] / 200
    
            if j % 5 == 0:
                s_out.sendto(np.ascontiguousarray(F).tobytes(), ("127.0.0.1", 50006))
        F += get_assistance_force(model.opt.wind)

        if data.ncon == 0:
            if j % 5 == 0:
                s_out2.sendto(np.ascontiguousarray(F).tobytes(), ("127.0.0.1", 50007))
                
        diff = data.xpos[body_id][:2] - np.array([2.5, 3.5])
        aligned = abs(diff[0]) < 0.05 and abs(diff[1]) < 0.05
        if force_world[2] > 3000 and aligned:
            F = np.zeros(2)
            s_out.sendto(np.ascontiguousarray(F).tobytes(), ("127.0.0.1", 50006))
            break

        mujoco.mj_step(model, data)
        viewer.sync()
        

    s_in.close()
    s_out.close()
    s_out2.close()
