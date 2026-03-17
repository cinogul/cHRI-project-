import mujoco
import mujoco.viewer
import time
import numpy as np
import socket, struct

send_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # create a send socket
recv_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # create a receive socket
recv_sock.bind(("127.0.0.1", 40001)) # bind the socket to port 40002

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

    <body name="turbine_base_body" pos="8 8 0">
      <inertial pos="0 0 0" mass="20000" diaginertia="10000 10000 15000"/>
      <geom type="mesh" mesh="turbine_base" contype="1" conaffinity="1" rgba="1 1 1 1"/>
    </body>

    <body name="crane" pos="0 0 0">
      <inertial pos="0 0 0" mass="50000" diaginertia="20000 20000 30000"/>
      <joint name="crane_base_hinge" type="hinge" axis="0 0 1" damping="1000000" range="-90 90" limited="true"/>
      <camera name="crane_low_view" pos="0.5 0 10" mode="targetbody" target="payload"/>
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



model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
# data.qpos[0] = 0.0
# data.qpos[1] = 7.0


i = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.fixedcamid = 0
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    while viewer.is_running():
        i += 1
        step_start = time.time()
        
        # Angle
        # data.qpos[0] = 0
        
        # Depth of trolley
        # data.qpos[1] = 5.5 
        
        # Height
        # data.ctrl[0] = 0.0
            
        model.opt.wind = [50*np.sin(i/1000), 0, 0]
        
        
        mujoco.mj_step(model, data)
        viewer.sync()
        

