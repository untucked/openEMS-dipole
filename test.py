import os
import subprocess
from xml.etree.ElementTree import Element, SubElement, ElementTree

import sys
sys.path.insert(0, r"C:\Users\eylan\Documents\antenna_sim\openEMS-Project\CSXCAD\python")
sys.path.insert(0, r"C:\Users\eylan\Documents\antenna_sim\openEMS-Project\python")


def write_dipole_xml(filename="dipole.xml"):
    openems = Element("openEMS")

    # --- FDTD ---
    fdtd = SubElement(openems, "FDTD")
    SubElement(fdtd, "EndCriteria").text = "1e-5"
    SubElement(fdtd, "BoundaryCond").text = "PEC PEC PEC PEC PEC PEC"
    SubElement(fdtd, "Excite", {"Type": "Gaussian", "f0": "2.4e9", "fc": "0.5e9"})

    # --- Geometry ---
    cs = SubElement(openems, "ContinuousStructure", {"unit": "m"})
    mat = SubElement(cs, "Material", {"name": "PEC"})
    SubElement(mat, "Epsilon").text = "1"

    box = SubElement(cs, "Box", {"material": "PEC"})
    SubElement(box, "Start").text = "-0.025 0 0"
    SubElement(box, "Stop").text = "0.025 0.001 0.001"

    # --- RectilinearGrid (must be direct child of <openEMS>) ---
    grid = SubElement(openems, "RectilinearGrid", {"unit": "m"})
    axes = {
        "X": [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03],
        "Y": [-0.005, 0.005],
        "Z": [-0.005, 0.005]
    }
    for axis_label, coords in axes.items():
        axis = SubElement(grid, axis_label)
        for val in coords:
            SubElement(axis, "Coord").text = str(val)

    # --- Write XML ---
    tree = ElementTree(openems)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"✅ XML written to: {filename}")

def run_openems(xml_file, outdir="."):
    print(f"🚀 Running openEMS on: {xml_file}")
    # call: openEMS.exe <output-folder> <xml-file>
    try:
        subprocess.run(
            ["openEMS.exe", outdir, xml_file],
            check=True
        )
        print("🎉 openEMS simulation completed successfully!")
    except FileNotFoundError:
        print("⚠️ Error: openEMS.exe not found. Is it in your PATH?")
    except subprocess.CalledProcessError as e:
        print(f"🚨 openEMS failed with exit code {e.returncode}")

if __name__ == "__main__":
    write_dipole_xml()
    run_openems("dipole.xml")
