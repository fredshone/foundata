"""Minimal writer for the resource-indexed Nesstar container format, reverse
engineered from nesstar_converter's reader (see its _parse_resource_index /
_parse_resource_layouts / _parse_variable_directory / extract_block_resource_indexed).

Every variable is encoded as a plain fixed-width space-padded ASCII "char"
column (mode_code=0, value_format_code=0) -- this sidesteps the compact
numeric (nibble/uintN) and cstring encodings real Nesstar files sometimes
use, since the reader's char path round-trips any string value (including
leading-zero codes like "01") byte-for-byte.
"""

import struct

SLOT_SIZE = 160
RESOURCE_INDEX_RECORD_SIZE = 15
DESCRIPTOR_RECORD_SIZE = 26
HEADER_SIZE = 64

NESSTAR_MAGIC = b"NESSTART"
RESOURCE_INDEX_OFFSET_FIELD = 0x25
DATASET_COUNT_FIELD = 0x2B
DESCRIPTOR_RECORD_SIZE_FIELD = 0x2D
DESCRIPTOR_TABLE_RECORD_ID_FIELD = 0x2F


def _u32(n):
    return struct.pack("<I", n)


def _u48(n):
    return struct.pack("<I", n & 0xFFFFFFFF) + struct.pack(
        "<H", (n >> 32) & 0xFFFF
    )


def _u16(n):
    return struct.pack("<H", n)


def _directory_entry(entry_index, name, width_value, variable_id):
    buf = bytearray(SLOT_SIZE)
    buf[0:4] = _u32(entry_index)
    buf[5] = (
        0  # value_format_code -- not a COMPACT_FAMILIES code, forces char path
    )
    buf[15:19] = _u32(variable_id)
    name_bytes = name.encode("utf-16-le")
    if len(name_bytes) > 64:
        raise ValueError(f"variable name too long for directory slot: {name!r}")
    buf[63 : 63 + len(name_bytes)] = name_bytes
    buf[149] = width_value
    buf[159] = (
        0  # mode_code -- 0 (not 5=compact numeric, not 1=cstring) -> char path
    )
    return bytes(buf)


def build_nesstar_file(datasets: list[dict]) -> bytes:
    """datasets: [{"name": str, "nrecs": int, "variables": [{"name": str, "width": int, "values": [str, ...]}]}]

    Returns the full synthetic .Nesstar container bytes.
    """
    record_id = 1
    payload_chunks = []
    payload_offset = 0
    variable_record_ids = []  # per dataset, per variable

    for ds in datasets:
        nrecs = ds["nrecs"]
        ids_for_ds = []
        for var in ds["variables"]:
            width = var["width"]
            values = var["values"]
            if len(values) != nrecs:
                raise ValueError(
                    f"{ds['name']}.{var['name']}: {len(values)} values != nrecs={nrecs}"
                )
            col_bytes = bytearray(width * nrecs)
            for i, v in enumerate(values):
                enc = v.encode("ascii")
                if len(enc) > width:
                    raise ValueError(
                        f"{ds['name']}.{var['name']}: value {v!r} wider than column width {width}"
                    )
                col_bytes[i * width : i * width + len(enc)] = enc
                # pad with spaces (char decode strips them on read)
                col_bytes[i * width + len(enc) : (i + 1) * width] = b" " * (
                    width - len(enc)
                )
            payload_chunks.append(bytes(col_bytes))
            ids_for_ds.append((record_id, payload_offset, len(col_bytes)))
            payload_offset += len(col_bytes)
            record_id += 1
        variable_record_ids.append(ids_for_ds)

    payload_bytes = b"".join(payload_chunks)

    directory_chunks = []
    directory_offset = 0
    directory_record_ids = []  # per dataset
    for ds, ids_for_ds in zip(datasets, variable_record_ids):
        entries = []
        for i, (var, (var_rid, _, _)) in enumerate(
            zip(ds["variables"], ids_for_ds)
        ):
            entries.append(
                _directory_entry(i, var["name"], var["width"], var_rid)
            )
        blob = b"".join(entries)
        directory_chunks.append(blob)
        directory_record_ids.append((record_id, directory_offset, len(blob)))
        directory_offset += len(blob)
        record_id += 1

    directory_bytes = b"".join(directory_chunks)

    descriptor_chunks = []
    for i, (ds, (dir_rid, _, _)) in enumerate(
        zip(datasets, directory_record_ids)
    ):
        buf = bytearray(DESCRIPTOR_RECORD_SIZE)
        buf[0:4] = _u32(i)
        buf[4:8] = _u32(len(ds["variables"]))
        buf[8:12] = _u32(ds["nrecs"])
        buf[20:22] = _u16(SLOT_SIZE)  # per-variable directory entry size
        buf[22:26] = _u32(dir_rid)
        descriptor_chunks.append(bytes(buf))
    descriptor_bytes = b"".join(descriptor_chunks)
    descriptor_table_record_id = record_id
    record_id += 1

    # ---- absolute offsets ----
    payload_start = HEADER_SIZE
    directories_start = payload_start + len(payload_bytes)
    descriptor_start = directories_start + len(directory_bytes)
    resource_index_start = descriptor_start + len(descriptor_bytes)

    index_records = []
    for ids_for_ds in variable_record_ids:
        for rid, off, length in ids_for_ds:
            index_records.append((rid, payload_start + off, length))
    for rid, off, length in directory_record_ids:
        index_records.append((rid, directories_start + off, length))
    index_records.append(
        (descriptor_table_record_id, descriptor_start, len(descriptor_bytes))
    )

    resource_index_bytes = _u32(len(index_records))
    for rid, off, length in index_records:
        rec = bytearray(RESOURCE_INDEX_RECORD_SIZE)
        rec[0:4] = _u32(rid)
        rec[4:10] = _u48(off)
        rec[10:14] = _u32(length)
        resource_index_bytes += bytes(rec)

    header = bytearray(HEADER_SIZE)
    header[0:8] = NESSTAR_MAGIC
    header[RESOURCE_INDEX_OFFSET_FIELD : RESOURCE_INDEX_OFFSET_FIELD + 6] = (
        _u48(resource_index_start)
    )
    header[DATASET_COUNT_FIELD] = len(datasets)
    header[DESCRIPTOR_RECORD_SIZE_FIELD : DESCRIPTOR_RECORD_SIZE_FIELD + 2] = (
        _u16(DESCRIPTOR_RECORD_SIZE)
    )
    header[
        DESCRIPTOR_TABLE_RECORD_ID_FIELD : DESCRIPTOR_TABLE_RECORD_ID_FIELD + 4
    ] = _u32(descriptor_table_record_id)

    return (
        bytes(header)
        + payload_bytes
        + directory_bytes
        + descriptor_bytes
        + resource_index_bytes
    )
