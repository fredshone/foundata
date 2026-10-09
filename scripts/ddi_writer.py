"""Minimal DDI XML builder matching what nesstar_converter.parse_ddi expects:
one <fileDscr> per dataset (ID + URI containing "Name=<table>", nested
<dimensns><caseQnty>), and a flat list of <var> elements (name/files/
location/varFormat/labl) tying each variable to its dataset's fileDscr ID.
"""

import xml.etree.ElementTree as ET


def build_ddi(datasets: list[dict]) -> bytes:
    """datasets: [{"fid": str, "name": str, "nrecs": int,
    "variables": [{"name": str, "width": int, "label": str}]}]
    """
    root = ET.Element("codeBook")
    file_section = ET.SubElement(root, "fileDscr_section")
    data_section = ET.SubElement(root, "dataDscr")

    for ds in datasets:
        fd = ET.SubElement(file_section, "fileDscr")
        fd.set("ID", ds["fid"])
        fd.set("URI", f"urn:nesstar:synthetic;Name={ds['name']}")
        dims = ET.SubElement(fd, "dimensns")
        cc = ET.SubElement(dims, "caseQnty")
        cc.text = str(ds["nrecs"])

        for var in ds["variables"]:
            v = ET.SubElement(data_section, "var")
            v.set("name", var["name"])
            v.set("files", ds["fid"])
            loc = ET.SubElement(v, "location")
            loc.set("width", str(var["width"]))
            vf = ET.SubElement(v, "varFormat")
            vf.set("type", "character")
            vf.set("dcml", "0")
            labl = ET.SubElement(v, "labl")
            labl.text = var.get("label", var["name"])

    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root)
