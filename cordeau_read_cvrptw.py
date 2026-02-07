import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# Notes (CVRPTW-only):
# - This module now supports ONLY Cordeau type=4 (VRPTW with a single depot), i.e. CVRPTW.
# - Coordinates are treated as Euclidean; output EDGE_WEIGHT_TYPE : EUC_2D.
# - Time windows and service times are exported to VRPLIB sections:
#     * SERVICE_TIME_SECTION
#     * TIME_WINDOW_SECTION
#   (These are the sections understood by many VRPTW readers, including PyVRP.)
# - Periodic/combination fields (f, a, list) are parsed but ignored (CVRPTW is non-periodic).
# - Validation is strict: malformed files raise CordeauFormatError.

class CordeauFormatError(ValueError):
    pass


def _read_nonempty_noncomment_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            lines.append(s)
    return lines


def _parse_cordeau_vrptw_node_line(line: str) -> Tuple[int, float, float, float, int, float, float]:
    """Parse one Cordeau VRPTW node line.

    Expected layout (Cordeau):
        i x y d q f a list e l

    Where `list` has length `a`. For CVRPTW, typically f/a/list are unused, but they may
    still be present (often as zeros).

    Returns:
        (i, x, y, service_duration_d, demand_q, tw_earliest_e, tw_latest_l)
    """
    toks = line.split()
    if len(toks) < 5:
        raise CordeauFormatError(f"Node line must contain at least 'i x y d q', got: {line!r}")

    i = int(float(toks[0]))
    x = float(toks[1])
    y = float(toks[2])
    d = float(toks[3])
    q = int(float(toks[4]))

    # If there are no more fields, treat as no TW (0, 0)
    if len(toks) == 5:
        return i, x, y, d, q, 0.0, 0.0

    # Try to parse f a [list...] e l, but be tolerant if some fields are missing.
    # Minimum to have TWs is ... e l (2 tokens).
    # Common case: i x y d q 0 0 e l   (no list tokens)
    # or:          i x y d q 0 0 0 e l (one dummy list token)
    # Official:    i x y d q f a <a ints> e l
    if len(toks) < 7:
        # Not enough tokens to reliably extract e/l; fall back to (0,0)
        return i, x, y, d, q, 0.0, 0.0

    # Parse f and a if present
    try:
        f = int(float(toks[5]))
        a = int(float(toks[6]))
    except Exception:
        # If format deviates, try reading last two as e/l
        try:
            e = float(toks[-2]); l = float(toks[-1])
            return i, x, y, d, q, e, l
        except Exception as e2:
            raise CordeauFormatError(f"Could not parse VRPTW node line: {line!r}") from e2

    # Skip visit-combination list of length a (may be empty)
    idx = 7 + max(a, 0)
    if len(toks) >= idx + 2:
        e = float(toks[idx])
        l = float(toks[idx + 1])
        return i, x, y, d, q, e, l

    # If not enough tokens, try last two
    if len(toks) >= 2:
        try:
            e = float(toks[-2]); l = float(toks[-1])
            return i, x, y, d, q, e, l
        except Exception:
            pass

    return i, x, y, d, q, 0.0, 0.0


def convert_cordeau_to_vrplib(input_path: str, output_path: str=None) -> str:
    """Convert a *Cordeau CVRPTW* instance (type=4) to a VRPLIB VRPTW file.

    Cordeau requirements enforced:
      - First line: "type m n t" with type=4
      - t must be 1 (single depot/day for CVRPTW)
      - Node lines are expected to cover indices 0..n (depot + n customers)

    VRPLIB output:
      - TYPE : VRPTW
      - VEHICLES : m
      - CAPACITY : Q (from the single D Q line)
      - NODE_COORD_SECTION
      - DEMAND_SECTION
      - SERVICE_TIME_SECTION
      - TIME_WINDOW_SECTION
      - DEPOT_SECTION (single depot id = 1, ending with -1)

    Returns the path to the written VRPLIB file.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(input_path)

    lines = _read_nonempty_noncomment_lines(p)
    if not lines:
        raise CordeauFormatError("Empty file.")

    # --- header: "type m n t"
    try:
        head_tokens = lines[0].split()
        if len(head_tokens) != 4:
            raise ValueError
        type_, m, n, t = map(int, head_tokens)
    except Exception as e:
        raise CordeauFormatError("First line must be: 'type m n t' with 4 integers.") from e

    if type_ != 4:
        raise NotImplementedError(f"Only CVRPTW/VRPTW instances are supported (Cordeau type=4). Got type={type_}.")

    if t != 1:
        raise NotImplementedError(f"CVRPTW is expected to have t=1 in Cordeau format. Got t={t}.")

    # --- next t lines: "D Q"  (we use only Q)
    if len(lines) < 1 + t:
        raise CordeauFormatError("Not enough lines for the D Q block.")
    toks = lines[1].split()
    if len(toks) < 2:
        raise CordeauFormatError("Expected 'D Q' on the line after the header.")
    # D is read but ignored
    _D = float(toks[0])
    capacity = int(float(toks[1]))
    vehicles = m

    # --- node lines: expected depot + n customers => n + 1 lines, indices 0..n
    remain = lines[1 + t:]
    expected = n + 1
    if len(remain) < expected:
        raise CordeauFormatError(f"Expected at least {expected} node lines, got {len(remain)}.")

    node_lines = remain[:expected]
    parsed = [_parse_cordeau_vrptw_node_line(ln) for ln in node_lines]

    # Validate depot line index
    dep_i, dep_x, dep_y, dep_d, dep_q, dep_e, dep_l = parsed[0]
    if dep_i != 0:
        raise CordeauFormatError(f"For CVRPTW, first node line should be the depot with i=0. Got i={dep_i}.")

    # Build VRPLIB blocks
    name = p.stem
    out_type = "VRPTW"
    dimension = n + 1

    # VRPLIB node order: depot as 1, customers as 2..n+1
    nodes: List[Tuple[int, float, float]] = [(1, dep_x, dep_y)]
    demands: List[Tuple[int, int]] = [(1, 0)]
    service_times: List[Tuple[int, float]] = [(1, 0.0 if dep_d is None else float(dep_d))]
    time_windows: List[Tuple[int, float, float]] = [(1, float(dep_e), float(dep_l))]

    # Customers (cordeau i=1..n)
    for k, (i, x, y, d, q, e, l) in enumerate(parsed[1:], start=2):
        # Be strict about ids if present
        if i != k - 1:
            # Many datasets are consistent; if not, still accept but warn via strict check:
            # Raise to avoid silent misalignment.
            raise CordeauFormatError(
                f"Customer id mismatch: expected Cordeau i={k-1} for customer line {k}, got i={i}. "
                "(If your instance uses a different numbering, adjust this check.)"
            )
        nodes.append((k, x, y))
        demands.append((k, int(q)))
        service_times.append((k, float(d)))
        time_windows.append((k, float(e), float(l)))

    dep_ids = [1]

    # --- Compose VRPLIB text
    vrplib_lines: List[str] = []
    vrplib_lines.append(f"NAME : {name}")
    vrplib_lines.append(f"TYPE : {out_type}")
    vrplib_lines.append(f"DIMENSION : {dimension}")
    vrplib_lines.append(f"VEHICLES : {vehicles}")
    vrplib_lines.append(f"CAPACITY : {capacity}")
    vrplib_lines.append("EDGE_WEIGHT_TYPE : EUC_2D")

    vrplib_lines.append("NODE_COORD_SECTION")
    for idx, x, y in nodes:
        ix = int(x) if abs(x - int(x)) < 1e-9 else x
        iy = int(y) if abs(y - int(y)) < 1e-9 else y
        vrplib_lines.append(f"{idx} {ix} {iy}")

    vrplib_lines.append("DEMAND_SECTION")
    for idx, q in demands:
        vrplib_lines.append(f"{idx} {int(q)}")

    vrplib_lines.append("SERVICE_TIME_SECTION")
    for idx, d in service_times:
        # keep as integer if integral
        dd = int(d) if abs(d - int(d)) < 1e-9 else d
        vrplib_lines.append(f"{idx} {dd}")

    vrplib_lines.append("TIME_WINDOW_SECTION")
    for idx, e, l in time_windows:
        ee = int(e) if abs(e - int(e)) < 1e-9 else e
        ll = int(l) if abs(l - int(l)) < 1e-9 else l
        vrplib_lines.append(f"{idx} {ee} {ll}")

    vrplib_lines.append("DEPOT_SECTION")
    for dep in dep_ids:
        vrplib_lines.append(str(dep))
    vrplib_lines.append("-1")
    vrplib_lines.append("EOF")

    text = "\n".join(vrplib_lines) + "\n"

    # --- Write file
    if output_path is None:
        output_path = str(p.with_suffix(".vrptw.vrp"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


def convert_vrplib_to_cordeau(input_path: str, output_path: str = None) -> str:
    """Convert a VRPLIB VRPTW instance into a Cordeau CVRPTW (type=4) text file.

    Supported VRPLIB assumptions:
      - TYPE: "VRPTW" (or "CVRPTW" if present; treated as VRPTW)
      - Single depot in DEPOT_SECTION
      - NODE_COORD_SECTION present
      - DEMAND_SECTION present
      - SERVICE_TIME_SECTION present (optional; defaults to 0)
      - TIME_WINDOW_SECTION present (optional; defaults to 0 0)
      - CAPACITY and VEHICLES present

    Cordeau output:
      - First line: "4 m n 1"
      - Second line: "0 Q" (D=0, Q=CAPACITY)
      - Node lines 0..n (depot then customers):
          i x y d q f a e l   (with f=0, a=0, and no visit-combination list)
        If you prefer the fully explicit format, you can post-process to add an empty list marker.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(input_path)

    with open(p, "r", encoding="utf-8") as f:
        raw = [ln.rstrip("\n") for ln in f]

    def find_line(startswith: str) -> Optional[int]:
        for i, ln in enumerate(raw):
            if ln.strip().upper().startswith(startswith):
                return i
        return None

    def read_scalar_after(prefix: str, cast=int):
        idx = find_line(prefix)
        if idx is None:
            raise CordeauFormatError(f"Missing header line starting with '{prefix}'.")
        try:
            return cast(raw[idx].split(":")[1].strip())
        except Exception as e:
            raise CordeauFormatError(f"Malformed header: '{raw[idx]}'") from e

    type_idx = find_line("TYPE")
    if type_idx is None:
        raise CordeauFormatError("Missing TYPE line.")
    type_token = raw[type_idx].split(":")[1].strip().upper()
    if type_token not in ("VRPTW", "CVRPTW"):
        raise NotImplementedError(f"Only TYPE VRPTW/CVRPTW supported. Got '{type_token}'.")

    dimension = int(read_scalar_after("DIMENSION"))
    vehicles = int(read_scalar_after("VEHICLES"))
    capacity = int(read_scalar_after("CAPACITY"))

    # DEPOT_SECTION (single depot)
    dep_idx = find_line("DEPOT_SECTION")
    if dep_idx is None:
        raise CordeauFormatError("Missing DEPOT_SECTION.")
    depots: List[int] = []
    i = dep_idx + 1
    while i < len(raw):
        s = raw[i].strip()
        if not s:
            i += 1
            continue
        if s.upper().startswith("EOF"):
            break
        try:
            val = int(s.split()[0])
        except Exception:
            break
        if val == -1:
            break
        depots.append(val)
        i += 1
    if len(depots) != 1:
        raise CordeauFormatError(f"VRPTW->Cordeau expects exactly one depot, found {len(depots)}.")

    depot_id = depots[0]

    # NODE_COORD_SECTION
    nc_idx = find_line("NODE_COORD_SECTION")
    if nc_idx is None:
        raise CordeauFormatError("Missing NODE_COORD_SECTION.")
    coords: Dict[int, Tuple[float, float]] = {}
    j = nc_idx + 1
    while j < len(raw):
        s = raw[j].strip()
        if not s or s.upper().startswith(("DEMAND_SECTION", "SERVICE_TIME_SECTION", "TIME_WINDOW_SECTION", "DEPOT_SECTION", "EOF")):
            break
        toks = s.split()
        if len(toks) < 3:
            raise CordeauFormatError(f"Malformed NODE_COORD_SECTION line: '{s}'")
        idx = int(float(toks[0])); x = float(toks[1]); y = float(toks[2])
        coords[idx] = (x, y)
        j += 1

    # DEMAND_SECTION
    dem_idx = find_line("DEMAND_SECTION")
    if dem_idx is None:
        raise CordeauFormatError("Missing DEMAND_SECTION.")
    demand: Dict[int, int] = {}
    k = dem_idx + 1
    while k < len(raw):
        s = raw[k].strip()
        if not s or s.upper().startswith(("NODE_COORD_SECTION", "SERVICE_TIME_SECTION", "TIME_WINDOW_SECTION", "DEPOT_SECTION", "EOF")):
            break
        toks = s.split()
        if len(toks) < 2:
            raise CordeauFormatError(f"Malformed DEMAND_SECTION line: '{s}'")
        idx = int(float(toks[0])); q = int(float(toks[1]))
        demand[idx] = q
        k += 1

    # SERVICE_TIME_SECTION (optional)
    service: Dict[int, float] = {idx: 0.0 for idx in coords.keys()}
    st_idx = find_line("SERVICE_TIME_SECTION")
    if st_idx is not None:
        sidx = st_idx + 1
        while sidx < len(raw):
            s = raw[sidx].strip()
            if not s or s.upper().startswith(("TIME_WINDOW_SECTION", "DEPOT_SECTION", "EOF", "NODE_COORD_SECTION", "DEMAND_SECTION")):
                break
            toks = s.split()
            if len(toks) < 2:
                raise CordeauFormatError(f"Malformed SERVICE_TIME_SECTION line: '{s}'")
            idx = int(float(toks[0])); d = float(toks[1])
            service[idx] = d
            sidx += 1

    # TIME_WINDOW_SECTION (optional)
    tw: Dict[int, Tuple[float, float]] = {idx: (0.0, 0.0) for idx in coords.keys()}
    tw_idx = find_line("TIME_WINDOW_SECTION")
    if tw_idx is not None:
        tidx = tw_idx + 1
        while tidx < len(raw):
            s = raw[tidx].strip()
            if not s or s.upper().startswith(("DEPOT_SECTION", "EOF", "NODE_COORD_SECTION", "DEMAND_SECTION", "SERVICE_TIME_SECTION")):
                break
            toks = s.split()
            if len(toks) < 3:
                raise CordeauFormatError(f"Malformed TIME_WINDOW_SECTION line: '{s}'")
            idx = int(float(toks[0])); e = float(toks[1]); l = float(toks[2])
            tw[idx] = (e, l)
            tidx += 1

    # Validation
    all_ids = set(coords.keys())
    if set(demand.keys()) != all_ids:
        raise CordeauFormatError("Mismatch between NODE_COORD_SECTION and DEMAND_SECTION indices.")
    if depot_id not in all_ids:
        raise CordeauFormatError(f"Depot id {depot_id} not present in NODE_COORD_SECTION.")
    if demand.get(depot_id, 0) != 0:
        raise CordeauFormatError(f"Depot id {depot_id} must have demand 0.")

    if dimension != len(all_ids):
        raise CordeauFormatError(f"DIMENSION={dimension} does not match number of nodes={len(all_ids)}.")

    # Compose Cordeau CVRPTW (type=4)
    customer_ids = sorted(idx for idx in all_ids if idx != depot_id)
    n = len(customer_ids)

    header = f"4 {vehicles} {n} 1"
    dq_line = f"0 {capacity}"

    # Output node numbering required by Cordeau for VRP-like cases: depot=0, customers=1..n
    lines_out: List[str] = [header, dq_line]

    # Depot
    dx, dy = coords[depot_id]
    de, dl = tw.get(depot_id, (0.0, 0.0))
    dd = service.get(depot_id, 0.0)
    # i x y d q f a e l  (f=0, a=0)
    lines_out.append(f"0 {dx} {dy} {dd} 0 0 0 {de} {dl}")

    # Customers
    for new_i, old_idx in enumerate(customer_ids, start=1):
        x, y = coords[old_idx]
        q = demand[old_idx]
        d = service.get(old_idx, 0.0)
        e, l = tw.get(old_idx, (0.0, 0.0))
        lines_out.append(f"{new_i} {x} {y} {d} {q} 0 0 {e} {l}")

    if output_path is None:
        output_path = str(p.with_suffix(".cordeau_vrptw.txt"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")
    return output_path
