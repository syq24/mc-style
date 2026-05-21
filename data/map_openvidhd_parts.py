import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map filtered OpenVidHD video names to the required OpenVidHD zip parts."
    )
    parser.add_argument(
        "--input",
        default="/root/autodl-tmp/mc-style/data/OpenVidHD_30k_people_scene.csv",
        help="Filtered CSV containing at least a video column.",
    )
    parser.add_argument(
        "--json",
        default="/root/autodl-tmp/mc-style/data/OpenVidHD.json",
        help="OpenVidHD JSON mapping file.",
    )
    parser.add_argument(
        "--output-prefix",
        default="/root/autodl-tmp/mc-style/data/OpenVidHD_30k_people_scene_parts",
        help="Prefix for output files.",
    )
    return parser.parse_args()


def load_part_mapping(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        raw_parts = json.load(file)

    video_to_part = {}
    for part_entry in raw_parts:
        for part_name, file_names in part_entry.items():
            zip_name = f"OpenVidHD_{part_name.replace('part', 'part_')}.zip"
            for file_name in file_names:
                video_to_part[file_name] = {
                    "part_name": part_name,
                    "zip_name": zip_name,
                }
    return video_to_part


def main():
    args = parse_args()

    input_path = Path(args.input)
    json_path = Path(args.json)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "video" not in df.columns:
        raise ValueError("Input CSV must contain a 'video' column.")

    video_to_part = load_part_mapping(json_path)

    mapped_rows = []
    missing_videos = []

    for _, row in df.iterrows():
        video_name = row["video"]
        mapping = video_to_part.get(video_name)
        if mapping is None:
            missing_videos.append(video_name)
            continue

        mapped_rows.append(
            {
                "bucket": row["bucket"] if "bucket" in row else None,
                "video": video_name,
                "part_name": mapping["part_name"],
                "zip_name": mapping["zip_name"],
                "seconds": row["seconds"] if "seconds" in row else None,
                "caption": row["caption"] if "caption" in row else None,
            }
        )

    mapped_df = pd.DataFrame(mapped_rows)
    if mapped_df.empty:
        raise ValueError("No videos from the filtered CSV were found in OpenVidHD.json.")

    summary_df = (
        mapped_df.groupby(["part_name", "zip_name"], as_index=False)
        .agg(
            matched_videos=("video", "count"),
            people_videos=("bucket", lambda series: int((series == "people").sum())),
            generic_scene_videos=("bucket", lambda series: int((series == "generic_scene").sum())),
        )
        .sort_values(by=["matched_videos", "part_name"], ascending=[False, True])
    )

    zip_list_path = output_prefix.with_name(output_prefix.name + "_zip_list.txt")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    mapped_path = output_prefix.with_name(output_prefix.name + "_mapped.csv")
    missing_path = output_prefix.with_name(output_prefix.name + "_missing.txt")

    summary_df.to_csv(summary_path, index=False)
    mapped_df.to_csv(mapped_path, index=False)
    zip_list_path.write_text("\n".join(summary_df["zip_name"].tolist()) + "\n", encoding="utf-8")
    missing_path.write_text("\n".join(missing_videos) + ("\n" if missing_videos else ""), encoding="utf-8")

    print(f"Input rows: {len(df)}")
    print(f"Mapped rows: {len(mapped_df)}")
    print(f"Missing rows: {len(missing_videos)}")
    print(f"Required zip parts: {len(summary_df)}")
    print(f"Summary CSV: {summary_path}")
    print(f"Mapped CSV: {mapped_path}")
    print(f"Zip list: {zip_list_path}")
    print(f"Missing list: {missing_path}")

    print("Top 10 required parts by matched videos:")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()