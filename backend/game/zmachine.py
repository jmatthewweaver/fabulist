"""
Z-machine v3 game-file decoder.

Reads object descriptions directly from the binary game file — no dfrotz required.
In ZIL-compiled games (Zork, Hitchhiker's Guide, etc.) property 17 = P?LDESC and
property 11 = P?FDESC.  Both hold ROUTINE packed addresses (not raw string addresses).
The description text lives as an inline Z-string immediately after the first PRINT
opcode (0xB2) inside the routine.
"""
import struct
from pathlib import Path

# --- Alphabets (z-chars 6-31 index into these) ---
_A0 = "abcdefghijklmnopqrstuvwxyz"
_A1 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Index 0 = placeholder (z=6 in A2 is a 10-bit ZSCII escape, handled separately)
# Index 1 = newline, 2-11 = '0'-'9', 12+ = punctuation
_A2 = "\x00\n0123456789.,!?_#'\"/\\-:()"
_ALPHABETS = (_A0, _A1, _A2)

# ZIL property numbers
PROP_LDESC = 17   # P?LDESC — long/examine description (routine)
PROP_FDESC = 11   # P?FDESC — first-look description (routine)
_PRINT_OPCODE = 0xB2   # Z-machine v3 OP0:2 PRINT — prints inline z-string


# ---------------------------------------------------------------------------
# String decoder
# ---------------------------------------------------------------------------

def _decode_zstring(data: bytes, offset: int, abbreviations: list[str]) -> str:
    """Decode one Z-machine v3 encoded string at byte offset into data."""
    zchars: list[int] = []
    pos = offset
    while pos + 1 < len(data):
        word = (data[pos] << 8) | data[pos + 1]
        pos += 2
        zchars += [(word >> 10) & 0x1F, (word >> 5) & 0x1F, word & 0x1F]
        if word & 0x8000:       # high bit of last word signals end of string
            break

    out: list[str] = []
    alpha = 0   # current alphabet (0=A0, 1=A1, 2=A2)
    i = 0
    while i < len(zchars):
        z = zchars[i]; i += 1
        if z == 0:
            out.append(' '); alpha = 0
        elif z <= 3:
            # Abbreviation: the next z-char is the table index
            if i < len(zchars):
                idx = 32 * (z - 1) + zchars[i]; i += 1
                if 0 <= idx < len(abbreviations):
                    out.append(abbreviations[idx])
            alpha = 0
        elif z == 4:
            alpha = 1       # shift to A1
        elif z == 5:
            alpha = 2       # shift to A2
        else:
            # z >= 6: character from current alphabet
            if alpha == 2 and z == 6:
                # 10-bit ZSCII literal from next two z-chars
                if i + 1 < len(zchars):
                    out.append(chr((zchars[i] << 5) | zchars[i + 1])); i += 2
            else:
                t = _ALPHABETS[alpha]
                char_idx = z - 6
                if 0 <= char_idx < len(t):
                    out.append(t[char_idx])
            alpha = 0       # shifts are always single-character
    return ''.join(out)


def _is_plausible(text: str) -> bool:
    """Return False if text looks like mis-decoded routine bytes rather than prose."""
    if len(text) < 20 or len(text) > 3000:
        return False
    printable = sum(1 for c in text if c.isprintable() or c in ('\n', '\t'))
    if printable / len(text) < 0.90:
        return False
    # Genuine descriptions always start with a capital letter.
    # Garbage from false-positive 0xB2 hits inside opcode operands or encoded
    # strings typically starts with a space or lowercase — skip those so the
    # scan continues to the next 0xB2 candidate.
    return text.lstrip()[:1].isupper()


# ---------------------------------------------------------------------------
# Abbreviation table
# ---------------------------------------------------------------------------

def _read_abbreviations(data: bytes) -> list[str]:
    """
    Parse all 96 abbreviation strings from the file.
    The abbreviation pointer table is at address stored in header bytes 0x18-0x19.
    Each entry is a 2-byte word address (multiply by 2 for byte offset).
    Abbreviations may not themselves contain abbreviations, so we pass [].
    """
    table_addr = struct.unpack_from(">H", data, 0x18)[0]
    result: list[str] = []
    for i in range(96):
        word_addr = struct.unpack_from(">H", data, table_addr + i * 2)[0]
        result.append(_decode_zstring(data, word_addr * 2, []))
    return result


# ---------------------------------------------------------------------------
# Routine scanner: find PRINT opcode and decode its inline string
# ---------------------------------------------------------------------------

def _extract_print_string(
    data: bytes,
    routine_packed: int,
    abbreviations: list[str],
    name_hint: str = "",
) -> str:
    """
    Treat routine_packed as a ZIL routine packed address (byte_addr = packed * 2).
    Scan the routine body for PRINT opcodes (0xB2) and collect all plausible
    inline Z-strings found within the first 200 bytes of the routine body.

    Selection strategy:
      1. If any candidate contains a word from name_hint (≥ 4 chars), return the
         first such candidate.  This filters false-positive 0xB2 hits that
         accidentally decode another object's description.
      2. Otherwise fall back to the first plausible candidate.
    """
    byte_addr = routine_packed * 2
    if byte_addr + 2 >= len(data):
        return ""

    num_locals = data[byte_addr]
    if num_locals > 15:
        return ""       # not a valid ZIL routine header

    body_start = byte_addr + 1 + num_locals * 2
    end = min(body_start + 200, len(data) - 2)

    candidates: list[str] = []
    for i in range(body_start, end):
        if data[i] == _PRINT_OPCODE:
            try:
                text = _decode_zstring(data, i + 1, abbreviations)
                if _is_plausible(text):
                    candidates.append(text)
            except Exception:
                pass

    if not candidates:
        return ""

    # Prefer a candidate that mentions the object's own name
    if name_hint:
        hint_words = [w.lower() for w in name_hint.split() if len(w) >= 4]
        if hint_words:
            for c in candidates:
                c_lower = c.lower()
                if any(w in c_lower for w in hint_words):
                    return c

    return candidates[0]


# ---------------------------------------------------------------------------
# Property table reader
# ---------------------------------------------------------------------------

def _get_routine_prop(
    data: bytes,
    prop_table_addr: int,
    abbreviations: list[str],
    prop_num: int,
    name_hint: str = "",
) -> str:
    """
    Scan an object's property table for prop_num, treat the 2-byte value
    as a routine packed address, and extract the description via PRINT scan.
    Properties are stored in strictly descending order.
    """
    name_words = data[prop_table_addr]
    pos = prop_table_addr + 1 + name_words * 2      # skip object name z-string

    while pos < len(data):
        sb = data[pos]
        if sb == 0:
            break                           # end-of-properties sentinel
        num = sb & 0x1F
        size = (sb >> 5) + 1               # data size in bytes (1-8)
        pos += 1

        if num == prop_num:
            if size == 2:
                packed = struct.unpack_from(">H", data, pos)[0]
                return _extract_print_string(data, packed, abbreviations, name_hint)
            break                           # found but unexpected size
        elif num < prop_num:
            break                           # descending order: target not present
        pos += size

    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def routine_props_by_id(
    game_path: str,
    prop_nums: tuple[int, ...] | None = (PROP_LDESC, PROP_FDESC),
) -> dict[int, dict[int, int]]:
    """
    For every object, return its 2-byte property values that point at routines:
        {obj_id: {prop_num: packed_routine_address, ...}, ...}

    `prop_nums` filters which properties to collect (default: LDESC=17, FDESC=11);
    pass None to collect every 2-byte property (wider candidate net).

    The packed value is a ZIL routine packed address — byte address = value * 2.
    This is the id-keyed join key for txd-sourced description candidates (Step 2):
    object id -> routine address -> txd's decoded PRINT strings.
    """
    data = Path(game_path).read_bytes()
    if data[0] != 3:
        raise ValueError(f"Only Z-machine v3 is supported (this file is v{data[0]})")

    obj_table = struct.unpack_from(">H", data, 0x0A)[0]
    first_obj = obj_table + 31 * 2
    OBJ_ENTRY = 9

    result: dict[int, dict[int, int]] = {}
    for obj_id in range(1, 256):
        entry = first_obj + (obj_id - 1) * OBJ_ENTRY
        if entry + OBJ_ENTRY > len(data):
            break
        prop_addr = struct.unpack_from(">H", data, entry + 7)[0]
        if prop_addr == 0 or prop_addr >= len(data):
            break

        name_words = data[prop_addr]
        if name_words > 20:
            break       # walked off the object table into garbage

        props: dict[int, int] = {}
        pos = prop_addr + 1 + name_words * 2
        while pos < len(data):
            sb = data[pos]
            if sb == 0:
                break
            num = sb & 0x1F
            size = (sb >> 5) + 1
            pos += 1
            if size == 2 and (prop_nums is None or num in prop_nums):
                props[num] = struct.unpack_from(">H", data, pos)[0]
            pos += size

        if props:
            result[obj_id] = props

    return result


def _read_flat_objects(
    data: bytes,
    abbreviations: list[str],
    ldesc_prop: int,
    fdesc_prop: int,
) -> list[dict]:
    """
    Parse all object-table entries and return a flat list of dicts:
        {id, name, description, parent_id, sibling_id, child_id}
    Stops when object entries give way to garbage (name_words > 20).
    """
    obj_table = struct.unpack_from(">H", data, 0x0A)[0]
    first_obj  = obj_table + 31 * 2     # skip 31 default-property words
    OBJ_ENTRY  = 9

    objects: list[dict] = []
    for obj_id in range(1, 256):
        entry = first_obj + (obj_id - 1) * OBJ_ENTRY
        if entry + OBJ_ENTRY > len(data):
            break

        parent_id  = data[entry + 4]
        sibling_id = data[entry + 5]
        child_id   = data[entry + 6]
        prop_addr  = struct.unpack_from(">H", data, entry + 7)[0]

        if prop_addr == 0 or prop_addr >= len(data):
            break

        name_words = data[prop_addr]
        if name_words == 0:
            name = ""
        elif name_words > 20:
            break
        else:
            name = _decode_zstring(data, prop_addr + 1, abbreviations)

        desc = _get_routine_prop(data, prop_addr, abbreviations, ldesc_prop, name)
        if not desc:
            desc = _get_routine_prop(data, prop_addr, abbreviations, fdesc_prop, name)

        objects.append({
            "id":         obj_id,
            "name":       name,
            "description": desc,
            "parent_id":  parent_id,
            "sibling_id": sibling_id,
            "child_id":   child_id,
        })

    return objects


def extract_world(
    game_path: str,
    ldesc_prop: int = PROP_LDESC,
    fdesc_prop: int = PROP_FDESC,
) -> dict[str, dict]:
    """
    Build a name-keyed world map from a Z-machine v3 game file.

    Returns:
        {
          "West of House": {
              "id":          64,
              "description": "You are standing in an open field...",
              "parent":      "outdoors",      # parent object's short name (or "" for roots)
              "children":    ["mailbox", "white house"],   # initial game-state contents
          },
          "mailbox": {
              "id":          230,
              "description": "...",
              "parent":      "West of House",
              "children":    ["leaflet"],
          },
          ...
        }

    Parent/child relationships reflect the game's *initial* state.
    Only Z-machine version 3 is supported; raises ValueError otherwise.
    """
    data = Path(game_path).read_bytes()
    version = data[0]
    if version != 3:
        raise ValueError(f"Only Z-machine v3 is supported (this file is v{version})")

    abbreviations = _read_abbreviations(data)
    flat = _read_flat_objects(data, abbreviations, ldesc_prop, fdesc_prop)

    # Index by id for quick lookup
    by_id: dict[int, dict] = {o["id"]: o for o in flat}

    # Build children list for each object using the child→sibling linked list
    children_of: dict[int, list[str]] = {}
    for obj in flat:
        kids: list[str] = []
        cid = obj["child_id"]
        seen: set[int] = set()
        while cid != 0 and cid in by_id and cid not in seen:
            kids.append(by_id[cid]["name"])
            seen.add(cid)
            cid = by_id[cid]["sibling_id"]
        children_of[obj["id"]] = kids

    # Build name-keyed world dict
    world: dict[str, dict] = {}
    for obj in flat:
        if not obj["name"]:
            continue
        parent_name = by_id[obj["parent_id"]]["name"] if obj["parent_id"] in by_id else ""
        world[obj["name"]] = {
            "id":          obj["id"],
            "description": obj["description"],
            "parent":      parent_name,
            "children":    children_of[obj["id"]],
        }

    return world


def extract_descriptions(
    game_path: str,
    ldesc_prop: int = PROP_LDESC,
    fdesc_prop: int = PROP_FDESC,
) -> dict[str, str]:
    """Convenience wrapper — returns {name: description} without relationship data."""
    world = extract_world(game_path, ldesc_prop, fdesc_prop)
    return {name: entry["description"] for name, entry in world.items() if entry["description"]}
