import argparse
import math
from pathlib import Path

from PIL import Image, UnidentifiedImageError


DEFAULT_MAX_PIXELS = 399360
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strategy B: downscale images so width*height <= max_pixels while preserving aspect ratio."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input image root directory (recursive).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root directory. If omitted, images are rewritten in place.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=f"Maximum pixel area. Default: {DEFAULT_MAX_PIXELS}",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality when writing .jpg/.jpeg.",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=6,
        help="PNG compression level [0-9].",
    )
    parser.add_argument(
        "--keep-mtime",
        action="store_true",
        help="Keep original file modified time when writing in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be resized, do not write files.",
    )
    return parser.parse_args()


def iter_images(root_dir: Path):
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS:
            yield path


def target_size(width: int, height: int, max_pixels: int):
    area = width * height
    if area <= max_pixels:
        return width, height, 1.0

    scale = math.sqrt(max_pixels / float(area))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    while new_w * new_h > max_pixels:
        if new_w >= new_h and new_w > 1:
            new_w -= 1
        elif new_h > 1:
            new_h -= 1
        else:
            break

    return new_w, new_h, scale


def save_image(image: Image.Image, out_path: Path, suffix: str, jpeg_quality: int, png_compress_level: int):
    suffix = suffix.lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix in {".jpg", ".jpeg"}:
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
        return

    if suffix == ".png":
        image.save(out_path, format="PNG", compress_level=png_compress_level, optimize=True)
        return

    if suffix == ".webp":
        image.save(out_path, format="WEBP", quality=95, method=6)
        return

    image.save(out_path)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir does not exist or is not a directory: {input_dir}")

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    scanned = 0
    resized = 0
    skipped = 0
    errors = 0

    for src_path in iter_images(input_dir):
        scanned += 1
        rel_path = src_path.relative_to(input_dir)
        dst_path = src_path if output_dir is None else output_dir / rel_path

        try:
            old_stat = src_path.stat() if (args.keep_mtime and output_dir is None) else None
            with Image.open(src_path) as img:
                width, height = img.size
                new_w, new_h, _ = target_size(width, height, args.max_pixels)

                if new_w == width and new_h == height:
                    skipped += 1
                    continue

                resized += 1
                print(f"[RESIZE] {rel_path}  {width}x{height} -> {new_w}x{new_h}")
                if args.dry_run:
                    continue

                resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                save_image(
                    resized_img,
                    dst_path,
                    src_path.suffix,
                    args.jpeg_quality,
                    args.png_compress_level,
                )

                if old_stat is not None:
                    mtime_ns = old_stat.st_mtime_ns
                    atime_ns = old_stat.st_atime_ns
                    dst_path.touch(exist_ok=True)
                    # Keep original access/modify times for reproducibility.
                    import os

                    os.utime(dst_path, ns=(atime_ns, mtime_ns))

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            errors += 1
            print(f"[ERROR] {rel_path}: {exc}")

        if scanned % 500 == 0:
            print(
                f"Progress: scanned={scanned}, resized={resized}, skipped={skipped}, errors={errors}"
            )

    print("\nDone.")
    print(f"Scanned: {scanned}")
    print(f"Resized: {resized}")
    print(f"Skipped (already <= max pixels): {skipped}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
