"""
Reliable description extraction via `txd` (ztools Z-machine disassembler).

Why txd: it parses every routine correctly and decodes inline PRINT / PRINT_RET
strings, so — unlike a raw scan for the PRINT opcode byte (0xB2) — it never mistakes
an operand byte for a PRINT. Every string it emits is a real, correctly-decoded game
string.

Join strategy (Step 2): the binary gives each object's description-routine *packed
address* via `zmachine.routine_props_by_id`. We run txd in address mode so each
routine header carries its byte address, then map object id -> routine -> inline
strings as description candidates.

NOTE: the exact address-mode header format is confirmed against a live `txd` run on
the server (see `_ROUTINE_HDR`). The PRINT-extraction below is validated against the
symbolic dump and is format-independent.
"""
import re
import subprocess

# Routine header. We accept an optional leading hex byte-address (address mode) and
# an optional Rnnnn label (symbolic mode), so the same parser handles both outputs.
#   "Routine R0255, 1 local (0000)"          -> rnum=0255, addr=None
#   "Main routine R0007, 0 locals ()"        -> rnum=0007
#   "<hexaddr>: Routine ..." / "Routine <hexaddr>: ..."  (address mode)
_ROUTINE_HDR = re.compile(
    r'^\s*(?:(?P<addr1>[0-9a-f]{4,6}):\s*)?'
    r'(?:Main\s+)?Routine\b'
    r'(?:\s+(?P<addr2>[0-9a-f]{4,6})\b)?'
    r'(?:.*?\bR(?P<rnum>[0-9a-f]+))?',
    re.IGNORECASE,
)

_PRINT_OP = re.compile(r'\b(?:PRINT|PRINT_RET)\s+"')


def _extract_strings_from_block(block: str) -> list[str]:
    """
    Pull every inline PRINT / PRINT_RET string from one routine's disassembly text,
    in order. Strings may wrap across source lines, so we collapse internal
    whitespace. A string ends at the next double-quote — txd renders a literal quote
    raw (it does not double it), but description text never contains double-quotes
    (only dialogue in action messages does), so first-quote termination is correct
    for our targets.
    """
    out: list[str] = []
    for m in _PRINT_OP.finditer(block):
        i = m.end()                      # just past the opening quote
        j = block.find('"', i)           # next quote closes it
        if j == -1:
            continue
        text = " ".join(block[i:j].split())   # collapse wrapped whitespace
        if text:
            out.append(text)
    return out


def parse_txd(text: str) -> dict:
    """
    Parse txd output into:
        {"by_addr": {routine_byte_addr: [strings]},
         "by_rnum": {rnum_int:        [strings]}}

    `by_addr` is populated only when txd emitted addresses (address mode) — that's
    the map used for the object join. `by_rnum` is always populated and is handy for
    testing against a symbolic dump.
    """
    by_addr: dict[int, list[str]] = {}
    by_rnum: dict[int, list[str]] = {}

    # Split into routine blocks on the header line. txd writes "Routine ..." and
    # "Main routine ..." at column 0 (note the lowercase 'r' in "Main routine").
    headers = list(re.finditer(r'(?im)^(?:Main\s+)?Routine\b.*$', text))
    for idx, h in enumerate(headers):
        m = _ROUTINE_HDR.match(h.group(0))
        if not m:
            continue
        start = h.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[start:end]
        strings = _extract_strings_from_block(block)
        if not strings:
            continue

        addr_hex = m.group("addr1") or m.group("addr2")
        if addr_hex:
            by_addr[int(addr_hex, 16)] = strings
        if m.group("rnum"):
            by_rnum[int(m.group("rnum"), 16)] = strings

    return {"by_addr": by_addr, "by_rnum": by_rnum}


def _run_txd(game_path: str, txd_path: str) -> str:
    # Flags finalized from the server probe; -n requests address output in ztools txd.
    result = subprocess.run(
        [txd_path, "-n", game_path],
        capture_output=True, text=True, timeout=60,
    )
    return result.stdout


def extract_candidates(
    game_path: str,
    prop_addrs_by_id: dict[int, dict[int, int]],
    txd_path: str = "txd",
) -> dict[int, list[str]]:
    """
    Return {obj_id: [clean candidate description strings]}.

    `prop_addrs_by_id` is from zmachine.routine_props_by_id():
    {obj_id: {prop_num: packed_routine_address}}. We convert each packed address to a
    byte address (×2) and look up the routine's inline strings in txd's address map.
    Strings are de-duplicated per object, preserving order (LDESC props first).
    """
    parsed = parse_txd(_run_txd(game_path, txd_path))
    by_addr = parsed["by_addr"]

    candidates: dict[int, list[str]] = {}
    for obj_id, props in prop_addrs_by_id.items():
        seen: set[str] = set()
        ordered: list[str] = []
        # LDESC (17) before FDESC (11) before any others
        for prop_num in sorted(props, key=lambda p: (p != 17, p != 11, p)):
            byte_addr = props[prop_num] * 2
            for s in by_addr.get(byte_addr, []):
                if s not in seen:
                    seen.add(s)
                    ordered.append(s)
        if ordered:
            candidates[obj_id] = ordered
    return candidates
