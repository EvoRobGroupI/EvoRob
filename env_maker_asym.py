import random
import mujoco
import mujoco_viewer  # you already have mujoco-python-viewer
import xml.etree.ElementTree as ET
import math
import time

# -----------------------------------------------------------
# Step 1: Parse and modify the Slider environment XML
# -----------------------------------------------------------
slider_tree = ET.parse("SliderEnv.xml")
slider_root = slider_tree.getroot()

# Locate the <worldbody> element in the SliderEnv
slider_worldbody = slider_root.find("worldbody")
if slider_worldbody is None:
    raise RuntimeError("No <worldbody> element found in SliderEnv.xml")

# (Optional) Remove any existing “pole” geoms in the SliderEnv
for geom in list(slider_worldbody.findall("geom")):
    if "pole" in geom.attrib.get("name", ""):
        slider_worldbody.remove(geom)

# Add a grid of poles (2 rows × 20 cols) to the SliderEnv
num_rows = 2
num_cols = 20
spacing_x = 5       # meters between poles along X
spacing_y = 3.14    # meters between poles along Y
jitter = 0        # max ±0.5 m random perturbation in X only

# You can adjust starts so that the grid is centered or offset as desired
x_start = -spacing_x
y_start = -spacing_y / 2

for row in range(num_rows):
    for col in range(num_cols):
        # base grid position
        x_center = x_start + col * spacing_x
        y = y_start + row * spacing_y

        # add a random perturbation in X
        x = x_center + random.uniform(0, jitter)

        # pole_body = ET.SubElement(slider_worldbody, "body")

        pole_body = ET.SubElement(
            slider_worldbody, 
            "body", 
            {"name": f"pole_{row}_{col}"}
        )

        ET.SubElement(
            pole_body,
            "geom",
            {
                "type": "cylinder",
                "pos": f"{x:.2f} {y:.2f} 1.2",
                "size": "0.25 2",
                "rgba": "1 0.3 0.3 1",
                "name": f"pole_{row}_{col}",
            },
        )

# -----------------------------------------------------------
# Step 2: Parse the Turtle XML and merge it into the Slider XML
# -----------------------------------------------------------
turtle_tree = ET.parse("Turtle.xml")
turtle_root = turtle_tree.getroot()

for turtle_child in list(turtle_root):
    tag = turtle_child.tag
    if tag == "worldbody":
        for elem in list(turtle_child):
            slider_worldbody.append(elem)
    elif tag in {"asset", "actuator", "sensor", "tendon", "contact", "compiler", "default", "visual"}:
        slider_root.append(turtle_child)
    else:
        slider_root.append(turtle_child)

# -----------------------------------------------------------
# Step 3: Write out the merged environment to a new XML file
# -----------------------------------------------------------
output_filename = "CombinedSliderTurtle.xml"
slider_tree.write(output_filename, encoding="utf-8", xml_declaration=True)
print(f"Wrote merged XML to {output_filename}")

# -----------------------------------------------------------
# Step 4: Load the merged XML into MuJoCo and launch the viewer
# -----------------------------------------------------------
model = mujoco.MjModel.from_xml_path(output_filename)
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

# -----------------------------------------------------------
# Step 5: (Optional) Drive any actuators if present
# -----------------------------------------------------------
def actuator_id(model, name):
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return None if idx == -1 else idx

desired_actuators = [
    "motor_shoulder_L",
    "motor_elbow_L",
    "motor_shoulder_R",
    "motor_elbow_R",
]
act_ids = [
    idx
    for name in desired_actuators
    if (idx := actuator_id(model, name)) is not None
]

t0 = time.time()
while viewer.is_alive:
    if act_ids:
        φ = 2 * math.pi * 0.6 * (time.time() - t0)  # 0.6 Hz oscillation
        data.ctrl[act_ids] = [
            0.9 * math.sin(φ),
            0.0,
            0.9 * math.sin(φ),
            0.0,
        ]
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
