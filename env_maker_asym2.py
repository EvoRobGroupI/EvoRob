"""
build_env.py – generate CombinedSliderTurtle.xml and launch MuJoCo viewer
Adds a 160° horizontal fan (±80°, step 8°) of 21 perfectly flat range-finders.
"""

import math, random, time, xml.etree.ElementTree as ET
import mujoco, mujoco_viewer

# ───────────────── 1. merge Slider + poles (unchanged) ─────────────────────
slider_tree = ET.parse("SliderEnv.xml")
slider_root = slider_tree.getroot()
worldbody   = slider_root.find("worldbody") or ET.SubElement(slider_root,"worldbody")
jitter = 0

for g in list(worldbody.findall("geom")):
    if "pole" in g.get("name", ""):
        worldbody.remove(g)

for r in range(2):
    for c in range(20):
        x = -4.5 + c*5 + random.uniform(-jitter, jitter)
        y = -1.57 + r*3.14
        body = ET.SubElement(worldbody, "body", name=f"pole_{r}_{c}")
        ET.SubElement(body, "geom",
            type="cylinder", pos=f"{x:.2f} {y:.2f} 1.2",
            size="0.25 2", rgba="1 0.3 0.3 1", name=f"pole_{r}_{c}")

# ───────────────── 2. merge Turtle.xml ─────────────────────────────────────
turtle_root = ET.parse("Turtle.xml").getroot()
for blk in list(turtle_root):
    if blk.tag == "worldbody":
        worldbody.extend(list(blk))
    else:
        slider_root.append(blk)

turtle_body = worldbody.find(".//body[@name='turtle']")
if turtle_body is None:
    raise RuntimeError("Turtle body not found")

# ──────────── 2a. add 21 *flat* ray sites ──────────────────────────────────
angles = list(range(-80, 81, 8))             # –80 … +80 deg
SQ2    = math.sqrt(2) / 2

for θ in angles:
    rad = math.radians(θ)
    s, c = math.sin(rad), math.cos(rad)

    quat = ( SQ2,            # w  = cos(45°)
            -SQ2 *  s,       # x
             SQ2 *  c,       # y
             0.0 )           # z
    quat_str = f"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.1f}"

    ET.SubElement(
        turtle_body, "site",
        name=f"ray_{θ:+d}", type="sphere",
        pos="0.9 0 -0.18",           # 90 cm ahead, 8 cm up
        quat=quat_str, size="0.001"
    )

# ──────────── 2b. matching range-finder sensors ────────────────────────────
sensor_block = slider_root.find("sensor") or ET.SubElement(slider_root, "sensor")
for θ in angles:
    ET.SubElement(sensor_block, "rangefinder",
                  name=f"rf_{θ:+d}", site=f"ray_{θ:+d}")

# ───────────────── 3. write XML ────────────────────────────────────────────
out_xml = "CombinedSliderTurtle.xml"
slider_tree.write(out_xml, encoding="utf-8", xml_declaration=True)
print("✔ wrote", out_xml)

# ───────────────── 4. load & viewer ────────────────────────────────────────
m = mujoco.MjModel.from_xml_path(out_xml); d = mujoco.MjData(m)
viewer = mujoco_viewer.MujocoViewer(m, d)
if hasattr(viewer.scn, "selsensor"):
    viewer.scn.selsensor = -1      # draw all beams

# ───────────────── 5. simple flipper demo ──────────────────────────────────
ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
       for n in ("motor_shoulder_L","motor_elbow_L",
                 "motor_shoulder_R","motor_elbow_R")]
ids = [i for i in ids if i != -1]

t0 = time.time()
while viewer.is_alive:
    if ids:
        φ = 2*math.pi*0.6*(time.time()-t0)
        d.ctrl[ids] = [0.9*math.sin(φ), 0, 0.9*math.sin(φ), 0]
    mujoco.mj_step(m, d)
    viewer.render()

viewer.close()
