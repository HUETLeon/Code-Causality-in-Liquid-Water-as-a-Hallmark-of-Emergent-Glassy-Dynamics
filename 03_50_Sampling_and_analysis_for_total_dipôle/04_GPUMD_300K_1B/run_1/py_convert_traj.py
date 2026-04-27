#!/usr/bin/env python3
"""
Stream-convert extended (LAMMPS-style) extended-XYZ trajectory to
GROMACS conf.gro + traj.xtc.

Features:
 - streaming (ase.io.iread) to avoid OOM
 - preserves box (Lattice=...) and Time= header per frame (if present)
 - uses MDAnalysis Universe.empty(...) and writes with XTCWriter.write(u.atoms)
 - argparse + optional stride + optional tqdm progress bar

Requirements:
    pip install ase MDAnalysis
(Optionally) pip install tqdm

Notes:
 - This script assumes Time= in your extended-XYZ header is numeric (units: keep as-is),
   convert if needed (e.g. fs->ps) by adding a conversion factor.
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import ase.io
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter

def parse_args():
    p = argparse.ArgumentParser(description="Convert extended-XYZ -> conf.gro + traj.xtc (streaming)")
    p.add_argument("xyz", help="Input extended-XYZ file (extended format with Lattice= and optional Time=)")
    p.add_argument("--gro", default="conf.gro", help="Output GRO filename (default: conf.gro)")
    p.add_argument("--xtc", default="traj.xtc", help="Output XTC filename (default: traj.xtc)")
    p.add_argument("--stride", type=int, default=1, help="Write every Nth frame (default 1 = every frame)")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="Multiply Time header by this factor (useful to convert fs->ps: 0.001). Default 1.0 (no change).")
    return p.parse_args()

def xyz_to_gro_xtc_stream(xyz_file, gro_file="conf.gro", xtc_file="traj.xtc", stride=1, time_scale=1.0):
    if stride < 1:
        raise ValueError("stride must be >= 1")

    # optional progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except Exception:
        use_tqdm = False

    reader = ase.io.iread(xyz_file, index=":")
    try:
        first = next(reader)
    except StopIteration:
        raise RuntimeError("No frames found in input file.")

    n_atoms = len(first)
    symbols = first.get_chemical_symbols()
    print(f"Input: {xyz_file} -- first frame has {n_atoms} atoms")

    # Build minimal Universe: 1 residue (common for unknown topology)
    u = mda.Universe.empty(n_atoms, n_residues=1, trajectory=True)
    # name/type per-atom
    u.add_TopologyAttr("name", symbols)
    u.add_TopologyAttr("type", symbols)
    # single residue/segment for simplicity
    u.add_TopologyAttr("resid", [1])
    u.add_TopologyAttr("resname", ["MOL"])
    u.add_TopologyAttr("segid", ["SYS"])

    # Write initial structure via ASE (conf.gro). This yields a valid GRO file
    print(f"Writing structure (first frame) -> {gro_file}")
    first.write(gro_file)

    # Prepare reusable timestep attached to the universe
    ts = u.trajectory.ts

    def frame_time_from_info(frame, fallback_index):
        # ASE stores parsed extended-xyz headers in frame.info (keys vary by parser)
        # Typical key: "Time" or "time" or sometimes value in frame.info as string
        t = None
        for key in ("Time", "time"):
            if key in frame.info:
                try:
                    t = float(frame.info[key])
                    break
                except Exception:
                    # sometimes ASE stores non-numeric; ignore
                    t = None
        if t is None:
            t = float(fallback_index)
        # apply scaling if user requested (e.g., fs->ps multiply by 0.001)
        return t * time_scale

    def set_ts_from_frame(frame, frame_index):
        # ASE positions are in Å; convert to nm for XTC
        ts.positions = frame.get_positions() / 10.0
        # ASE cell.array is 3x3 in Å; convert to nm
        # For orthorhombic boxes this still works; MDAnalysis will accept triclinic_dimensions
        ts.triclinic_dimensions = np.array(frame.cell.array, dtype=float) / 10.0
        ts.time = frame_time_from_info(frame, frame_index)

    # open writer (streaming)
    print(f"Opening XTC writer -> {xtc_file}")
    with XTCWriter(xtc_file, n_atoms=n_atoms) as W:
        # write first frame if it matches stride
        if 0 % stride == 0:
            set_ts_from_frame(first, 0)
            W.write(u.atoms)

        # iterate and write remaining frames
        it = enumerate(reader, start=1)
        if use_tqdm:
            # We can't know total frames in advance without scanning; tqdm will be indefinite
            it = tqdm(it, desc="Writing frames", unit="frame")

        written = 1 if (0 % stride == 0) else 0
        for idx, frame in it:
            # safety: atom count must match
            if len(frame) != n_atoms:
                raise RuntimeError(f"Frame {idx}: atom count {len(frame)} != {n_atoms}")

            if (idx % stride) != 0:
                continue

            set_ts_from_frame(frame, idx)
            W.write(u.atoms)   # pass AtomGroup/Universe, not a raw Timestep
            written += 1

            # occasional flush / small progress printed by tqdm if installed
            # (no need to print every frame)

    print(f"Done. Wrote {written} frames to {xtc_file} and structure to {gro_file}")

def main():
    args = parse_args()
    try:
        xyz_to_gro_xtc_stream(args.xyz, gro_file=args.gro, xtc_file=args.xtc,
                              stride=args.stride, time_scale=args.time_scale)
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise

if __name__ == "__main__":
    main()

