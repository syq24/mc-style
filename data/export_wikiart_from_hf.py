import argparse
import csv
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


DEFAULT_ROOT_DIR = "/root/autodl-tmp/mc-style/data/style_images/wikiart"
DEFAULT_RAW_GLOB = "data/*.parquet"

ARTIST_NAMES = [
    "Unknown Artist", "boris-kustodiev", "camille-pissarro", "childe-hassam", "claude-monet",
    "edgar-degas", "eugene-boudin", "gustave-dore", "ilya-repin", "ivan-aivazovsky",
    "ivan-shishkin", "john-singer-sargent", "marc-chagall", "martiros-saryan", "nicholas-roerich",
    "pablo-picasso", "paul-cezanne", "pierre-auguste-renoir", "pyotr-konchalovsky", "raphael-kirchner",
    "rembrandt", "salvador-dali", "vincent-van-gogh", "hieronymus-bosch", "leonardo-da-vinci",
    "albrecht-durer", "edouard-cortes", "sam-francis", "juan-gris", "lucas-cranach-the-elder",
    "paul-gauguin", "konstantin-makovsky", "egon-schiele", "thomas-eakins", "gustave-moreau",
    "francisco-goya", "edvard-munch", "henri-matisse", "fra-angelico", "maxime-maufra",
    "jan-matejko", "mstislav-dobuzhinsky", "alfred-sisley", "mary-cassatt", "gustave-loiseau",
    "fernando-botero", "zinaida-serebriakova", "georges-seurat", "isaac-levitan", "joaquã\xadn-sorolla",
    "jacek-malczewski", "berthe-morisot", "andy-warhol", "arkhip-kuindzhi", "niko-pirosmani",
    "james-tissot", "vasily-polenov", "valentin-serov", "pietro-perugino", "pierre-bonnard",
    "ferdinand-hodler", "bartolome-esteban-murillo", "giovanni-boldini", "henri-martin", "gustav-klimt",
    "vasily-perov", "odilon-redon", "tintoretto", "gene-davis", "raphael",
    "john-henry-twachtman", "henri-de-toulouse-lautrec", "antoine-blanchard", "david-burliuk", "camille-corot",
    "konstantin-korovin", "ivan-bilibin", "titian", "maurice-prendergast", "edouard-manet",
    "peter-paul-rubens", "aubrey-beardsley", "paolo-veronese", "joshua-reynolds", "kuzma-petrov-vodkin",
    "gustave-caillebotte", "lucian-freud", "michelangelo", "dante-gabriel-rossetti", "felix-vallotton",
    "nikolay-bogdanov-belsky", "georges-braque", "vasily-surikov", "fernand-leger", "konstantin-somov",
    "katsushika-hokusai", "sir-lawrence-alma-tadema", "vasily-vereshchagin", "ernst-ludwig-kirchner", "mikhail-vrubel",
    "orest-kiprensky", "william-merritt-chase", "aleksey-savrasov", "hans-memling", "amedeo-modigliani",
    "ivan-kramskoy", "utagawa-kuniyoshi", "gustave-courbet", "william-turner", "theo-van-rysselberghe",
    "joseph-wright", "edward-burne-jones", "koloman-moser", "viktor-vasnetsov", "anthony-van-dyck",
    "raoul-dufy", "frans-hals", "hans-holbein-the-younger", "ilya-mashkov", "henri-fantin-latour",
    "m.c.-escher", "el-greco", "mikalojus-ciurlionis", "james-mcneill-whistler", "karl-bryullov",
    "jacob-jordaens", "thomas-gainsborough", "eugene-delacroix", "canaletto",
]

GENRE_NAMES = [
    "abstract_painting", "cityscape", "genre_painting", "illustration", "landscape",
    "nude_painting", "portrait", "religious_painting", "sketch_and_study", "still_life", "Unknown Genre",
]

STYLE_NAMES = [
    "Abstract_Expressionism", "Action_painting", "Analytical_Cubism", "Art_Nouveau", "Baroque",
    "Color_Field_Painting", "Contemporary_Realism", "Cubism", "Early_Renaissance", "Expressionism",
    "Fauvism", "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance", "Minimalism",
    "Naive_Art_Primitivism", "New_Realism", "Northern_Renaissance", "Pointillism", "Pop_Art",
    "Post_Impressionism", "Realism", "Rococo", "Romanticism", "Symbolism", "Synthetic_Cubism", "Ukiyo_e",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert locally downloaded WikiArt parquet files under raw/ into image files and metadata.csv/txt."
        )
    )
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR, help="WikiArt root directory containing raw/, images/, metadata/.")
    parser.add_argument("--raw-glob", default=DEFAULT_RAW_GLOB, help="Glob for parquet files under raw/.")
    parser.add_argument("--max-parquet-files", type=int, default=None, help="Optional limit on how many parquet files to process.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional global sample limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=64, help="Arrow batch size when iterating parquet files.")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg", help="Output image format.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when image-format is jpg.")
    parser.add_argument("--skip-existing-images", action="store_true", help="Do not rewrite images that already exist.")
    parser.add_argument("--write-txt", action="store_true", help="Also write a plain text metadata manifest.")
    return parser.parse_args()


def ensure_dirs(root_dir):
    root = Path(root_dir)
    raw_dir = root / "raw"
    images_dir = root / "images"
    metadata_dir = root / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return root, raw_dir, images_dir, metadata_dir


def find_parquet_files(raw_dir, raw_glob):
    parquet_files = sorted(raw_dir.glob(raw_glob))
    if parquet_files:
        return parquet_files

    # Backward-compatible fallback for users who keep parquet directly under raw/.
    if raw_glob == DEFAULT_RAW_GLOB:
        return sorted(raw_dir.glob("*.parquet"))

    return parquet_files


def decode_label(index, names):
    if index is None:
        return ""
    try:
        index = int(index)
    except (TypeError, ValueError):
        return str(index)
    if 0 <= index < len(names):
        return names[index]
    return str(index)


def image_bytes_from_cell(cell):
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return cell["bytes"]
    if hasattr(cell, "as_py"):
        python_value = cell.as_py()
        return image_bytes_from_cell(python_value)
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"Unsupported image cell type: {type(cell)}")


def save_image(image_bytes, output_path, image_format, jpeg_quality):
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        if image_format == "jpg":
            image.save(output_path, format="JPEG", quality=jpeg_quality)
        else:
            image.save(output_path, format="PNG")


def build_text(artist, genre, style):
    pieces = []
    if style:
        pieces.append(f"style: {style}")
    if artist:
        pieces.append(f"artist: {artist}")
    if genre:
        pieces.append(f"genre: {genre}")
    return "; ".join(pieces)


def iter_records(parquet_path, batch_size):
    parquet_file = pq.ParquetFile(parquet_path)
    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in record_batch.to_pylist():
            yield row


def main():
    args = parse_args()
    root_dir, raw_dir, images_dir, metadata_dir = ensure_dirs(args.root_dir)
    parquet_files = find_parquet_files(raw_dir, args.raw_glob)
    if args.max_parquet_files is not None:
        parquet_files = parquet_files[: args.max_parquet_files]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {raw_dir} matching {args.raw_glob}")

    metadata_csv_path = metadata_dir / "wikiart_metadata.csv"
    metadata_txt_path = metadata_dir / "wikiart_metadata.txt"

    print(f"Root dir: {root_dir}")
    print(f"Raw dir: {raw_dir}")
    print(f"Images dir: {images_dir}")
    print(f"Metadata CSV: {metadata_csv_path}")
    print(f"Parquet files: {len(parquet_files)}")
    if args.write_txt:
        print(f"Metadata TXT: {metadata_txt_path}")

    row_count = 0
    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["relpath", "filename", "artist", "genre", "style", "text", "source_dataset", "raw_file"],
        )
        writer.writeheader()

        txt_file = open(metadata_txt_path, "w", encoding="utf-8") if args.write_txt else None
        try:
            for parquet_index, parquet_path in enumerate(parquet_files):
                parquet_stem = parquet_path.stem
                print(f"Processing [{parquet_index + 1}/{len(parquet_files)}]: {parquet_path.name}")

                for local_index, row in enumerate(iter_records(parquet_path, args.batch_size)):
                    if args.max_samples is not None and row_count >= args.max_samples:
                        print(f"Reached max-samples={args.max_samples}")
                        print(f"Convert complete: {row_count} samples")
                        return

                    image_bytes = image_bytes_from_cell(row["image"])
                    artist = decode_label(row.get("artist"), ARTIST_NAMES)
                    genre = decode_label(row.get("genre"), GENRE_NAMES)
                    style = decode_label(row.get("style"), STYLE_NAMES)

                    extension = "jpg" if args.image_format == "jpg" else "png"
                    filename = f"{parquet_stem}_{local_index:06d}.{extension}"
                    output_path = images_dir / filename
                    relpath = output_path.relative_to(root_dir).as_posix()

                    if not (args.skip_existing_images and output_path.exists()):
                        save_image(image_bytes, output_path, args.image_format, args.jpeg_quality)

                    text = build_text(artist, genre, style)
                    metadata_row = {
                        "relpath": relpath,
                        "filename": filename,
                        "artist": artist,
                        "genre": genre,
                        "style": style,
                        "text": text,
                        "source_dataset": "wikiart",
                        "raw_file": parquet_path.name,
                    }
                    writer.writerow(metadata_row)
                    if txt_file is not None:
                        txt_file.write(f"{relpath}\t{text}\n")

                    row_count += 1
                    if row_count % 500 == 0:
                        print(f"Converted {row_count} samples...")
        finally:
            if txt_file is not None:
                txt_file.close()

    print(f"Convert complete: {row_count} samples")


if __name__ == "__main__":
    main()