import argparse
import os
from dotenv import load_dotenv

from cutter_pipeline.outline_openai import generate_outline_png
from cutter_pipeline.stl_dispatch import generate_stl_from_trace
from cutter_pipeline.trace_outline import trace_png_to_polygon

def main():
    load_dotenv()

    p = argparse.ArgumentParser(description="Outline -> Trace -> Cookie Cutter STL pipeline")
    p.add_argument("--prompt", help="Text prompt for outline generation (optional)")
    p.add_argument("--png", help="Existing outline PNG path (optional)")
    p.add_argument("--outdir", default="output")
    p.add_argument("--name", default="cookie_cutter")
    p.add_argument(
        "--topology",
        choices=["auto", "single", "lattice"],
        default="auto",
        help="Shape topology: auto-detect, single outline, or connected grid lattice",
    )

    p.add_argument("--width-mm", type=float, default=95.0)
    p.add_argument("--wall-mm", type=float, default=1.0, help="Base cutter wall thickness (minimum enforced: 0.45mm)")
    p.add_argument("--total-h-mm", type=float, default=25.0)
    p.add_argument("--flange-h-mm", type=float, default=7.226)
    p.add_argument("--flange-out-mm", type=float, default=5.0)
    p.add_argument("--bevel-h-mm", type=float, default=2.0, help="Outer taper height at top of cutter")
    p.add_argument(
        "--bevel-top-wall-mm",
        type=float,
        default=0.5,
        help="Target wall thickness at the top of the taper (minimum enforced: 0.45mm)",
    )
    p.add_argument("--cleanup-mm", type=float, default=0.5, help="Remove features smaller than this (0 disables)")
    p.add_argument("--keep-holes", action="store_true", help="Keep interior holes instead of filling them")
    p.add_argument("--min-component-area-mm2", type=float, default=25.0, help="Discard tiny disconnected islands below this area")
    p.add_argument("--threshold", type=int, default=200)
    p.add_argument("--simplify", type=float, default=0.002)
    p.add_argument("--smooth-radius", type=float, default=1.0, help="Gaussian blur radius (pixels) before tracing")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not args.png and not args.prompt:
        raise SystemExit("Provide either --png OR --prompt")

    png_path = args.png
    if args.prompt and not png_path:
        png_path = os.path.join(args.outdir, f"{args.name}.png")
        generate_outline_png(args.prompt, png_path)

    svg_path = os.path.join(args.outdir, f"{args.name}.svg")
    traced = trace_png_to_polygon(
        png_path,
        svg_path,
        threshold=args.threshold,
        simplify_epsilon=args.simplify,
        smooth_radius=args.smooth_radius,
        topology=args.topology,
    )

    stl_path = os.path.join(args.outdir, f"{args.name}.stl")
    generate_stl_from_trace(
        traced,
        stl_path,
        target_width_mm=args.width_mm,
        wall_mm=args.wall_mm,
        total_h_mm=args.total_h_mm,
        flange_h_mm=args.flange_h_mm,
        flange_out_mm=args.flange_out_mm,
        bevel_h_mm=args.bevel_h_mm,
        bevel_top_wall_mm=args.bevel_top_wall_mm,
        cleanup_mm=args.cleanup_mm,
        drop_holes=not args.keep_holes,
        min_component_area_mm2=args.min_component_area_mm2,
    )

    print("Wrote:")
    print(" PNG:", png_path)
    print(" SVG:", svg_path)
    print(" STL:", stl_path)
    print(" Topology:", traced.topology)
    if traced.topology == "lattice":
        print(f" Grid: {traced.cols} cols x {traced.rows} rows")

if __name__ == "__main__":
    main()
