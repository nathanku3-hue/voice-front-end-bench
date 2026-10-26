#!/usr/bin/env python3
"""Render a GTKWave-style waveform figure from the AXI4-Stream VCD dump.

The point of this script is that the figure is drawn *from the VCD*, not from
the simulator's own counters. If the handshake in the file were wrong, the
picture would show it; there is no second source of truth to disagree with.

    python scripts/render_vcd.py results/axi4s_backpressure.vcd \
        --out results/axi4s_waveform.png --zoom-cycle 1072
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DIGITAL = ["s_tvalid", "s_tready", "m_tvalid", "m_tready", "m_tlast", "m_tuser"]
ANALOG = ["fifo_level", "beat_in_burst"]


def parse_vcd(path: Path):
    """Return (signals{id: (name, width)}, changes{id: [(time, value)]})."""
    signals: dict[str, tuple[str, int]] = {}
    changes: dict[str, list[tuple[int, int]]] = {}
    time = 0
    var_re = re.compile(r"\$var\s+wire\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\[[^\]]+\])?\s+\$end")
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("$var"):
            match = var_re.match(line)
            if match:
                width, ident, name = match.groups()
                signals[ident] = (name, int(width))
                changes.setdefault(ident, [])
            continue
        if line.startswith("#"):
            time = int(line[1:])
            continue
        if line.startswith("$"):
            continue
        if line[0] in "01xz":
            ident = line[1:]
            if ident in changes:
                changes[ident].append((time, 1 if line[0] == "1" else 0))
            continue
        if line[0] in "bB":
            head, _, ident = line[1:].partition(" ")
            ident = ident.strip()
            if ident in changes:
                try:
                    changes[ident].append((time, int(head, 2)))
                except ValueError:
                    pass
    return signals, changes


def series(points, times):
    """Step-hold interpolation of one signal onto the given time grid."""
    if not points:
        return [0] * len(times)
    out = []
    index = 0
    current = 0
    for t in times:
        while index < len(points) and points[index][0] <= t:
            current = points[index][1]
            index += 1
        out.append(current)
    return out


def draw_digital(ax, times, values, base, height, colour):
    xs, ys = [], []
    previous = None
    for t, v in zip(times, values):
        y = base + (height if v else 0.0)
        if previous is None:
            xs.append(t)
            ys.append(y)
        elif y != previous:
            xs.extend([t, t])
            ys.extend([previous, y])
        previous = y
    if times:
        xs.append(times[-1])
        ys.append(previous if previous is not None else base)
    ax.plot(xs, ys, color=colour, lw=1.1, solid_joinstyle="miter")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vcd", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/axi4s_waveform.png"))
    parser.add_argument("--zoom-cycle", type=int, default=None,
                        help="centre of the zoom panel (default: first cycle with s_tready low)")
    parser.add_argument("--zoom-width", type=int, default=200)
    parser.add_argument("--period-ns", type=int, default=10,
                        help="VCD time step per cycle (100 MHz default); the plot axis is cycles")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals, raw_changes = parse_vcd(args.vcd)
    # The VCD time base is nanoseconds; the plot is in cycles.
    changes = {ident: [(t // args.period_ns, v) for t, v in points]
               for ident, points in raw_changes.items()}
    by_name = {name: ident for ident, (name, _) in signals.items()}
    if "clk" not in by_name:
        raise SystemExit("VCD has no clk signal")

    all_times = sorted({t for pts in changes.values() for t, _ in pts})
    if not all_times:
        raise SystemExit("VCD has no value changes")
    first, last = 0, max(all_times)
    grid = list(range(first, last + 1))
    times_ns = grid  # the x axis is cycles, not ns

    s_ready = series(changes.get(by_name.get("s_tready", ""), []), times_ns)
    if args.zoom_cycle is None:
        low = [c for c, v in zip(grid, s_ready) if v == 0]
        centre = low[0] if low else grid[len(grid) // 2]
    else:
        centre = args.zoom_cycle

    rows = DIGITAL + ANALOG
    height = 0.62
    row_y = {name: float(len(rows) - 1 - i) for i, name in enumerate(rows)}
    colours = {"s_tvalid": "#1f6feb", "s_tready": "#2f9e44", "m_tvalid": "#1f6feb",
               "m_tready": "#2f9e44", "m_tlast": "#f08c00", "m_tuser": "#d1495b",
               "fifo_level": "#7048e8", "beat_in_burst": "#0b7285"}

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(15.6, 7.4), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1.4, 1]))

    for ax, (lo, hi, title) in (
        (left, (first, last, f"overview: cycles {first}-{last}")),
        (right, (max(first, centre - args.zoom_width // 2),
                 min(last, centre + args.zoom_width // 2),
                 f"zoom: cycles {max(first, centre - args.zoom_width // 2)}-"
                 f"{min(last, centre + args.zoom_width // 2)}")),
    ):
        ts = [c for c in grid if lo <= c <= hi]
        if not ts:
            continue
        for name in DIGITAL:
            values = series(changes.get(by_name.get(name, ""), []), ts)
            draw_digital(ax, ts, values, row_y[name], height, colours[name])
        for name in ANALOG:
            values = series(changes.get(by_name.get(name, ""), []), ts)
            peak = max(1, max(values))
            scaled = [row_y[name] + v / peak * height for v in values]
            ax.fill_between(ts, row_y[name], scaled, step="post", color=colours[name],
                            alpha=0.30, lw=0)
            ax.plot(ts, scaled, drawstyle="steps-post", color=colours[name], lw=1.0)
            ax.axhline(row_y[name], color="#adb5bd", lw=0.6)
        # Shade the intervals where the downstream refused a beat.
        start = None
        for t, v in zip(ts, series(changes.get(by_name.get("m_tready", ""), []), ts)):
            if v == 0 and start is None:
                start = t
            elif v == 1 and start is not None:
                ax.axvspan(start, t, color="#d1495b", alpha=0.07, lw=0)
                start = None
        if start is not None:
            ax.axvspan(start, ts[-1], color="#d1495b", alpha=0.07, lw=0)
        ax.set_ylim(-0.4, len(rows) - 0.2)
        ax.set_yticks([row_y[name] + height / 2 for name in rows])
        ax.set_yticklabels(rows, fontsize=9, family="monospace")
        ax.set_xlim(ts[0], ts[-1])
        ax.set_xlabel("cycle @ 100 MHz")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", alpha=0.18, ls=":")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle(
        "AXI4-Stream handshake: downstream TREADY=0 freezes the pipeline instead of dropping beats",
        fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
