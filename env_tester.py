import mujoco
import mujoco_viewer               # you already have mujoco-python-viewer

# ------------------------------------------------------------------
# 0) tiny wrapper that keeps both XMLs intact
# ------------------------------------------------------------------
wrapper_xml = f"""
<mujoco model='slider_plus_turtle'>
  <include file='SliderEnv.xml'/>
  <include file='Turtle.xml'/>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(wrapper_xml)
data  = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

# ------------------------------------------------------------------
# 1) helper: name → id (returns None if the name is missing)
# ------------------------------------------------------------------
def actuator_id(model, name):
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return None if idx == -1 else idx

desired = ["motor_shoulder_L", "motor_elbow_L",
           "motor_shoulder_R", "motor_elbow_R"]

act_ids = [idx for name in desired
                 if (idx := actuator_id(model, name)) is not None]

# ------------------------------------------------------------------
# 2) loop – flap flippers if we found the motors
# ------------------------------------------------------------------
import math, time
t0 = time.time()
while viewer.is_alive:
    if act_ids:                         # drive only if motors exist
        φ = 2*math.pi*0.6*(time.time()-t0)   # 0.6 Hz stroke
        data.ctrl[act_ids] = [ 0.9*math.sin(φ),
                                0.6*math.sin(φ+math.pi/4),
                               -0.9*math.sin(φ),
                               -0.6*math.sin(φ+math.pi/4) ]
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
