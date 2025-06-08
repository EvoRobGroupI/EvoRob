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
spacing_x = 5  # meters between poles along X
spacing_y = 3.14  # meters between poles along Y
adjust_x = 0.5

# You can adjust starts so that the grid is centered or offset as desired
x_start = -spacing_x + adjust_x
y_start = -spacing_y / 2

for row in range(num_rows):
    for col in range(num_cols):
        x = x_start + col * spacing_x
        y = y_start + row * spacing_y
        pole_body = ET.SubElement(slider_worldbody, "body")
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

# We will merge all top-level sections from Turtle into the Slider root.
# Typically, a MuJoCo XML has sections like <asset>, <worldbody>, <actuator>, etc.
# For each section in Turtle.xml, append its children into the Slider root.

for turtle_child in list(turtle_root):
    tag = turtle_child.tag

    # If it's a <worldbody>, merge its child elements into the slider's worldbody.
    if tag == "worldbody":
        for elem in list(turtle_child):
            # Append each body/geom/etc from Turtle's worldbody
            slider_worldbody.append(elem)

    # If it's an <asset>, <actuator>, <sensor>, <tendon>, or other top-level
    # section, append it directly under the Slider root.
    elif tag in {"asset", "actuator", "sensor", "tendon", "contact", "asset", "compiler", "default", "visual"}:
        # Make sure we do not overwrite existing sections; simply append
        slider_root.append(turtle_child)

    # If there are other sections (e.g., <custom>), you can handle them similarly:
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
# Load the combined model from the newly written file
model = mujoco.MjModel.from_xml_path(output_filename)
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

# -----------------------------------------------------------
# Step 5: (Optional) Drive any actuators if present
# -----------------------------------------------------------
def actuator_id(model, name):
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return None if idx == -1 else idx

# Example: if the Turtle has motors called "motor_shoulder_L", etc.
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
        φ = 2 * math.pi * 0.4 * (time.time() - t0)  # 0.6 Hz oscillation
        data.ctrl[act_ids] = [
            math.sin(φ),
            0.0,
            math.sin(φ),
            0.0,
        ]
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
