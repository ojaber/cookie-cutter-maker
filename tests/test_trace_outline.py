from pathlib import Path

from PIL import Image, ImageDraw

from cutter_pipeline.trace_outline import trace_png_to_polygon


def _make_test_image(path: Path) -> None:
    """Create a simple black square on white background."""
    img = Image.new("L", (128, 128), color=255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([32, 32, 96, 96], fill=0)
    img.save(path)


def test_trace_png_to_polygon(tmp_path: Path) -> None:
    png_path = tmp_path / "square.png"
    svg_path = tmp_path / "square.svg"

    _make_test_image(png_path)

    result = trace_png_to_polygon(
        str(png_path),
        str(svg_path),
        threshold=200,
        simplify_epsilon=0.0005,
        smooth_radius=0.0,
    )

    # Basic validity checks
    assert result.topology == "single"
    assert result.polygon is not None
    assert result.polygon.is_valid
    # Expect area close to the drawn square (0.5*0.5 = 0.25 in normalized coords)
    assert 0.20 <= result.polygon.area <= 0.30
    assert svg_path.exists()


def test_trace_with_smoothing(tmp_path: Path) -> None:
    png_path = tmp_path / "square.png"
    svg_path = tmp_path / "square.svg"

    _make_test_image(png_path)

    result = trace_png_to_polygon(
        str(png_path),
        str(svg_path),
        threshold=200,
        simplify_epsilon=0.002,
        smooth_radius=1.0,
    )

    assert result.polygon.is_valid
    assert svg_path.exists()
