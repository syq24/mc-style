import argparse
import csv
import os
import requests
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_SUMMARY_CSV = "/root/autodl-tmp/mc-style/data/OpenVidHD_30k_people_scene_parts_summary.csv"
DEFAULT_MAPPED_CSV = "/root/autodl-tmp/mc-style/data/OpenVidHD_30k_people_scene_parts_mapped.csv"
DEFAULT_CONTENT_DIR = "/root/autodl-tmp/mc-style/data/content_videos"
DEFAULT_WORK_DIR = "/root/autodl-tmp/mc-style/data/.openvidhd_work"
DEFAULT_REPO_ID = "nkp37/OpenVid-1M"
DEFAULT_REPO_SUBDIR = "OpenVidHD"
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
DEFAULT_REVISION = "main"
DEFAULT_SPLIT_START = 15
DEFAULT_SPLIT_SUFFIXES = "aa,ab"
DEFAULT_LOG_DIR = "/root/autodl-tmp/mc-style/data/logs"
DEFAULT_CONNECT_TIMEOUT = 20
DEFAULT_READ_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_BACKOFF = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download OpenVidHD parts one by one, extract them, and keep only videos "
            "present in the filtered mapping."
        )
    )
    parser.add_argument(
        "--summary-csv",
        default=DEFAULT_SUMMARY_CSV,
        help="Summary CSV produced by map_openvidhd_parts.py.",
    )
    parser.add_argument(
        "--mapped-csv",
        default=DEFAULT_MAPPED_CSV,
        help="Mapped CSV produced by map_openvidhd_parts.py.",
    )
    parser.add_argument(
        "--rank-range",
        default="1-10",
        help="1-based inclusive range over the summary ranking, for example 1-10 or 11-20.",
    )
    parser.add_argument(
        "--content-dir",
        default=DEFAULT_CONTENT_DIR,
        help="Directory where kept videos will be stored.",
    )
    parser.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
        help="Temporary working directory for downloads and extraction.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--repo-subdir",
        default=DEFAULT_REPO_SUBDIR,
        help="Subdirectory inside the dataset repo that stores OpenVidHD archives.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Base endpoint used for resolve URLs, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Dataset revision to download from.",
    )
    parser.add_argument(
        "--split-start",
        type=int,
        default=DEFAULT_SPLIT_START,
        help="Parts with numeric id >= split-start are treated as split archives.",
    )
    parser.add_argument(
        "--split-suffixes",
        default=DEFAULT_SPLIT_SUFFIXES,
        help="Comma-separated suffixes for split archives, for example aa,ab.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for run logs and per-part missing lists.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=DEFAULT_CONNECT_TIMEOUT,
        help="HTTP connect timeout in seconds.",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=DEFAULT_READ_TIMEOUT,
        help="HTTP read timeout in seconds for streaming downloads.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for each remote file download.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=int,
        default=DEFAULT_RETRY_BACKOFF,
        help="Base backoff in seconds between download retries.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite videos already present in content-dir.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download remote archive pieces even if the cache already has them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve selected parts and remote files without downloading or extracting.",
    )
    return parser.parse_args()


def parse_rank_range(rank_range):
    text = rank_range.strip()
    if "-" not in text:
        value = int(text)
        return value, value

    start_text, end_text = text.split("-", 1)
    start = int(start_text)
    end = int(end_text)
    if start < 1 or end < start:
        raise ValueError(f"Invalid rank range: {rank_range}")
    return start, end


def load_selection(summary_csv, mapped_csv, rank_range):
    summary_df = pd.read_csv(summary_csv)
    mapped_df = pd.read_csv(mapped_csv)

    required_summary_columns = {"part_name", "zip_name", "matched_videos"}
    required_mapped_columns = {"part_name", "video"}
    if not required_summary_columns.issubset(summary_df.columns):
        missing = sorted(required_summary_columns - set(summary_df.columns))
        raise ValueError(f"Summary CSV missing columns: {missing}")
    if not required_mapped_columns.issubset(mapped_df.columns):
        missing = sorted(required_mapped_columns - set(mapped_df.columns))
        raise ValueError(f"Mapped CSV missing columns: {missing}")

    start_rank, end_rank = parse_rank_range(rank_range)
    selected_summary = summary_df.iloc[start_rank - 1 : end_rank].copy()
    if selected_summary.empty:
        raise ValueError("Selected rank range resolved to zero parts.")

    keep_videos_by_part = {}
    for part_name in selected_summary["part_name"]:
        part_rows = mapped_df[mapped_df["part_name"] == part_name]
        keep_videos_by_part[part_name] = set(part_rows["video"].astype(str))

    return selected_summary, keep_videos_by_part


def build_resolve_url(endpoint, repo_id, revision, remote_file):
    endpoint = endpoint.rstrip("/")
    return f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{remote_file}"


def parse_split_suffixes(split_suffixes):
    items = [item.strip() for item in split_suffixes.split(",") if item.strip()]
    if not items:
        raise ValueError("split-suffixes must contain at least one suffix.")
    return items


def resolve_remote_files(repo_subdir, part_name, split_start, split_suffixes):
    if not part_name.startswith("part"):
        raise ValueError(f"Unexpected part name: {part_name}")

    numeric_part = int(part_name.replace("part", "", 1))
    base_name = f"OpenVidHD_part_{numeric_part}"
    if numeric_part < split_start:
        return [f"{repo_subdir}/{base_name}.zip"]

    return [
        f"{repo_subdir}/{base_name}_part_{suffix}"
        for suffix in parse_split_suffixes(split_suffixes)
    ]


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def create_run_logging(log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / f"download_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "parts_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "rank_index",
                "part_name",
                "expected_keep_videos",
                "remote_files",
                "status",
                "kept_count",
                "skipped_existing",
                "missing_count",
                "missing_preview",
                "error",
            ],
        )
        writer.writeheader()
    return run_dir, csv_path


def append_part_log(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "rank_index",
                "part_name",
                "expected_keep_videos",
                "remote_files",
                "status",
                "kept_count",
                "skipped_existing",
                "missing_count",
                "missing_preview",
                "error",
            ],
        )
        writer.writerow(row)


def write_missing_videos(run_dir, part_name, missing_videos):
    missing_path = run_dir / f"{part_name}_missing.txt"
    missing_path.write_text(
        "\n".join(missing_videos) + ("\n" if missing_videos else ""),
        encoding="utf-8",
    )
    return missing_path


def assemble_archive(part_name, downloaded_paths, assemble_dir):
    assemble_dir.mkdir(parents=True, exist_ok=True)

    if len(downloaded_paths) == 1 and downloaded_paths[0].suffix == ".zip":
        return downloaded_paths[0]

    archive_path = assemble_dir / f"{part_name}.zip"
    with open(archive_path, "wb") as output_file:
        for segment_path in downloaded_paths:
            with open(segment_path, "rb") as input_file:
                shutil.copyfileobj(input_file, output_file, length=16 * 1024 * 1024)
    return archive_path


def download_file(url, destination, force_redownload, connect_timeout, read_timeout, max_retries, retry_backoff):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force_redownload:
        return destination

    partial_path = destination.with_name(destination.name + ".part")
    if force_redownload:
        if destination.exists():
            destination.unlink()
        if partial_path.exists():
            partial_path.unlink()

    attempt = 0
    while True:
        attempt += 1
        existing_size = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {}
        mode = "ab" if existing_size > 0 else "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        try:
            with requests.get(
                url,
                stream=True,
                timeout=(connect_timeout, read_timeout),
                headers=headers,
            ) as response:
                if existing_size > 0 and response.status_code == 200:
                    partial_path.unlink(missing_ok=True)
                    existing_size = 0
                    headers = {}
                    mode = "wb"
                    raise RuntimeError("Server did not honor range request; restarting download.")

                response.raise_for_status()
                with open(partial_path, mode) as output_file:
                    for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            output_file.write(chunk)
                            output_file.flush()

            partial_path.replace(destination)
            return destination
        except (requests.exceptions.RequestException, RuntimeError) as error:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to download {url} after {attempt} attempts: {error}"
                ) from error

            wait_seconds = retry_backoff * attempt
            print(
                f"Download attempt {attempt}/{max_retries} failed for {destination.name}: {error}. "
                f"Retrying in {wait_seconds}s..."
            )
            import time

            time.sleep(wait_seconds)


def download_remote_files(
    remote_files,
    endpoint,
    repo_id,
    revision,
    download_root,
    force_redownload,
    connect_timeout,
    read_timeout,
    max_retries,
    retry_backoff,
):
    local_paths = []
    for remote_file in remote_files:
        url = build_resolve_url(endpoint, repo_id, revision, remote_file)
        local_path = download_root / Path(remote_file).name
        download_file(
            url,
            local_path,
            force_redownload,
            connect_timeout,
            read_timeout,
            max_retries,
            retry_backoff,
        )
        local_paths.append(local_path)
    return local_paths


def extract_archive(archive_path, extract_dir):
    ensure_clean_dir(extract_dir)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)


def move_selected_videos(extract_dir, keep_videos, content_dir, overwrite_existing):
    kept_count = 0
    skipped_existing = 0
    missing_in_archive = set(keep_videos)

    for extracted_path in extract_dir.rglob("*"):
        if not extracted_path.is_file():
            continue

        file_name = extracted_path.name
        if file_name not in keep_videos:
            continue

        destination = content_dir / file_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite_existing:
            skipped_existing += 1
            missing_in_archive.discard(file_name)
            continue

        shutil.move(str(extracted_path), str(destination))
        kept_count += 1
        missing_in_archive.discard(file_name)

    return kept_count, skipped_existing, sorted(missing_in_archive)


def main():
    args = parse_args()

    summary_path = Path(args.summary_csv)
    mapped_path = Path(args.mapped_csv)
    content_dir = Path(args.content_dir)
    work_dir = Path(args.work_dir)
    log_dir = Path(args.log_dir)
    content_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    selected_summary, keep_videos_by_part = load_selection(
        summary_path,
        mapped_path,
        args.rank_range,
    )
    downloads_dir = work_dir / "downloads"
    assembled_dir = work_dir / "assembled"
    extracted_dir = work_dir / "extracted"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    assembled_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir, parts_log_csv = create_run_logging(log_dir)

    print(f"Selected rank range: {args.rank_range}")
    print(f"Selected logical parts: {len(selected_summary)}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Split start: {args.split_start}")
    print(f"Split suffixes: {args.split_suffixes}")
    print(f"Connect timeout: {args.connect_timeout}")
    print(f"Read timeout: {args.read_timeout}")
    print(f"Max retries: {args.max_retries}")
    print(f"Run log dir: {run_log_dir}")

    for index, summary_row in enumerate(selected_summary.itertuples(index=False), start=1):
        part_name = summary_row.part_name
        keep_videos = keep_videos_by_part[part_name]
        remote_files = resolve_remote_files(
            args.repo_subdir,
            part_name,
            args.split_start,
            args.split_suffixes,
        )

        print("=" * 80)
        print(
            f"[{index}/{len(selected_summary)}] {part_name} | keep {len(keep_videos)} videos | "
            f"remote files: {', '.join(Path(file_path).name for file_path in remote_files)}"
        )

        base_log_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "rank_index": index,
            "part_name": part_name,
            "expected_keep_videos": len(keep_videos),
            "remote_files": "|".join(Path(file_path).name for file_path in remote_files),
            "status": "pending",
            "kept_count": 0,
            "skipped_existing": 0,
            "missing_count": 0,
            "missing_preview": "",
            "error": "",
        }

        if args.dry_run:
            base_log_row["status"] = "dry_run"
            append_part_log(parts_log_csv, base_log_row)
            continue

        downloaded_paths = []
        archive_path = None
        try:
            downloaded_paths = download_remote_files(
                remote_files,
                args.endpoint,
                args.repo_id,
                args.revision,
                downloads_dir,
                args.force_redownload,
                args.connect_timeout,
                args.read_timeout,
                args.max_retries,
                args.retry_backoff,
            )
            archive_path = assemble_archive(part_name, downloaded_paths, assembled_dir)
            extract_archive(archive_path, extracted_dir)
            kept_count, skipped_existing, missing_videos = move_selected_videos(
                extracted_dir,
                keep_videos,
                content_dir,
                args.overwrite_existing,
            )

            print(
                f"Finished {part_name}: kept={kept_count}, skipped_existing={skipped_existing}, "
                f"missing_in_archive={len(missing_videos)}"
            )

            preview = ""
            if missing_videos:
                preview = ", ".join(missing_videos[:5])
                print(f"Missing sample preview: {preview}")
                write_missing_videos(run_log_dir, part_name, missing_videos)

            base_log_row.update(
                {
                    "status": "completed",
                    "kept_count": kept_count,
                    "skipped_existing": skipped_existing,
                    "missing_count": len(missing_videos),
                    "missing_preview": preview,
                }
            )
            append_part_log(parts_log_csv, base_log_row)
        except Exception as error:
            base_log_row.update(
                {
                    "status": "failed",
                    "error": str(error),
                }
            )
            append_part_log(parts_log_csv, base_log_row)
            raise
        finally:
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)
            if archive_path is not None and archive_path.exists() and archive_path.parent == assembled_dir:
                archive_path.unlink()
            for downloaded_path in downloaded_paths:
                if downloaded_path.exists():
                    downloaded_path.unlink()

    print("All selected parts processed.")


if __name__ == "__main__":
    main()