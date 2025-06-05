import mujoco
import mujoco_viewer               # you already have mujoco-python-viewer
import xml.etree.ElementTree as ET

# Load original MuJoCo XML
tree = ET.parse("SliderEnv.xml")
root = tree.getroot()

# Find the <worldbody> element
worldbody = root.find("worldbody")
if worldbody is None:
    raise RuntimeError("No <worldbody> element found in the XML.")

# Remove old poles (optional)
for geom in worldbody.findall("geom"):
    if "pole" in geom.attrib.get("name", ""):
        worldbody.remove(geom)

# Add a grid of poles (rows × cols)
num_rows = 2
num_cols = 20
spacing_x = 0.7  # meters between poles in X
spacing_y = 0.7  # meters between poles in Y

# x_start = -(num_cols - 1) / 2 * spacing_x
# y_start = -(num_rows - 1) / 2 * spacing_y
x_start = -spacing_x
y_start = -spacing_y/2

for row in range(num_rows):
    for col in range(num_cols):
        x = x_start + col * spacing_x
        y = y_start + row * spacing_y
        pole_body = ET.SubElement(worldbody, "body")
        ET.SubElement(pole_body, "geom", {
            "type": "cylinder",
            "pos": f"{x:.2f} {y:.2f} 0.5",
            "size": "0.05 0.5",
            "rgba": "1 0.3 0.3 1",
            "name": f"pole_{row}_{col}"
        })

# Save to a new file
tree.write("SliderEnv_inf.xml")
print("Wrote grid of poles to SliderEnv_inf.xml")



# ------------------------------------------------------------------
# 0) tiny wrapper that keeps both XMLs intact
# ------------------------------------------------------------------
wrapper_xml = f"""
<mujoco model='slider_plus_turtle'>
  <include file='SliderEnv_inf.xml'/>
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
        # φ = 2*math.pi*0.6*(time.time()-t0)   # 0.6 Hz stroke
        # data.ctrl[act_ids] = [ 0.9*math.sin(φ),
        #                         0.6*math.sin(φ+math.pi/4),
        #                        -0.9*math.sin(φ),
        #                        -0.6*math.sin(φ+math.pi/4) ]
        φ = 2*math.pi*0.6*(time.time()-t0)   # 0.6 Hz stroke
        data.ctrl[act_ids] = [ 0.9*math.sin(φ),
                                0.0,
                               0.9*math.sin(φ),
                                0 ]
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
