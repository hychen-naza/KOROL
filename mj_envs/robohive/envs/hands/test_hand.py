import mujoco
import matplotlib.pyplot as plt

# free_body_MJCF = """
# <mujoco>
#   <asset>
#     <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
#     rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
#     <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"
#     reflectance=".2"/>
#   </asset>

#   <worldbody>
#     <light pos="0 0 1" mode="trackcom"/>
#     <geom name="ground" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid" solimp=".99 .99 .01" solref=".001 1"/>
#     <body name="box_and_sphere" pos="0 0 0">
#       <freejoint/>
#       <geom name="red_box" type="box" size=".1 .1 .1" rgba="1 0 0 1" solimp=".99 .99 .01"  solref=".001 1"/>
#       <geom name="green_sphere" size=".06" pos=".1 .1 .1" rgba="0 1 0 1"/>
#       <camera name="fixed" pos="0 -.6 .3" xyaxes="1 0 0 0 1 2"/>
#       <camera name="track" pos="0 -.6 .3" xyaxes="1 0 0 0 1 2" mode="track"/>
#     </body>
#   </worldbody>
# </mujoco>
# """
# model = mujoco.MjModel.from_xml_string(free_body_MJCF)
# data = mujoco.MjData(model)
# height = 400
# width = 600

# with mujoco.Renderer(model, height, width) as renderer:
#   mujoco.mj_forward(model, data)
#   renderer.update_scene(data, "fixed")
#   image = renderer.render()

# # Display the image using matplotlib
# plt.imshow(image)
# plt.axis('off')  # Hide axes for better viewing
# plt.show()

import mujoco
import matplotlib.pyplot as plt
import mujoco_viewer
#import mujoco_py as mj
import numpy as np 
import pdb
import time
# Load the MuJoCo model from an XML file
# right_hand

scene_xml_path = './assets/DAPG_door.xml'#mujoco.MjModel.from_xml_path('./shadow_hand_soft/right_hand.xml')

# Load the scene model
model = mujoco.MjModel.from_xml_path(scene_xml_path)
data = mujoco.MjData(model)
pdb.set_trace()
# Initialize the viewer
# viewer = mujoco_viewer.MujocoViewer(model, data)

# viewer.vopt.geomgroup[0] = 1  # Enable collision group
# viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE


# Control loop for the hand to grasp the cube

time.sleep(3)
while True:

    # data.qpos[0] = 0
    # data.qpos[1] = 0
    # data.qpos[2] = 0.1
    # data.qpos[3] = 0
    # data.qpos[4] = 2**0.5 / 2
    # data.qpos[5] = 0
    # data.qpos[6] = 2**0.5 / 2

    # data.ctrl[3] = 0.5
    # data.ctrl[7] = 0.8
    # data.ctrl[10] = 0.8
    #data.ctrl[9] = 1.5
    # Step the simulation and render
    mujoco.mj_step(model, data)
    #force=data.sensor("contact_force").data
    #print("---print force: ", force) 
    #print(data.qpos)
    # print("---print force: ", data.qpos) 
    #viewer.render()
    #pdb.set_trace()
