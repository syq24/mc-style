import argparse
import re
from pathlib import Path

import pandas as pd


PEOPLE_PATTERN = re.compile(
    r"\b(person|people|man|woman|boy|girl|child|children|adult|male|female|human|face|hand|hands|portrait|couple|worker|chef|farmer|driver|dancer|skier|surfer|cyclist|runner|athlete|bride|groom)\b",
    re.IGNORECASE,
)

SCENE_PATTERN = re.compile(
    r"\b(street|road|city|town|village|building|room|kitchen|bedroom|office|park|garden|forest|mountain|beach|sea|ocean|river|lake|sky|cloud|sunset|snow|field|farm|restaurant|cafe|shop|market|living room|table|car|bus|train|boat|bridge|stadium)\b",
    re.IGNORECASE,
)

NEGATIVE_PATTERN = re.compile(
    r"\b(animation|anime|cartoon|illustration|cgi|3d render|rendered|game footage|gameplay|logo|text overlay|watermark|slide|presentation|meme|poster|screenshot)\b",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter OpenVidHD captions into people and generic-scene buckets."
    )
    parser.add_argument(
        "--input",
        default="/root/autodl-tmp/mc-style/data/OpenVidHD.csv",
        help="Path to the input OpenVidHD CSV.",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/mc-style/data/OpenVidHD_people_scene_filtered.csv",
        help="Path to the output filtered CSV.",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=4.0,
        help="Minimum clip duration in seconds.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=12.0,
        help="Maximum clip duration in seconds.",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=None,
        help="Optional cap for the people bucket after sorting.",
    )
    parser.add_argument(
        "--max-scene",
        type=int,
        default=None,
        help="Optional cap for the generic_scene bucket after sorting.",
    )
    return parser.parse_args()


def require_columns(df, required_columns):
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))


def safe_text(text):
    if pd.isna(text):
        return ""
    return str(text)


def extract_matches(text, pattern):
    return sorted(set(pattern.findall(safe_text(text))))


def compute_quality_score(df):
    score = pd.Series(0.0, index=df.index)
    if "aesthetic score" in df.columns:
        score = score + df["aesthetic score"].rank(pct=True, method="average") * 0.5
    if "temporal consistency score" in df.columns:
        score = score + df["temporal consistency score"].rank(pct=True, method="average") * 0.35
    if "motion score" in df.columns:
        score = score + df["motion score"].rank(pct=True, method="average") * 0.15
    return score


def finalize_bucket(df, bucket_name, max_items):
    if max_items is not None:
        df = df.head(max_items)
    df = df.copy()
    df["bucket"] = bucket_name
    return df


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    require_columns(df, ["video", "caption", "seconds"])

    df = df.copy()
    df["caption"] = df["caption"].map(safe_text)
    df["people_matches"] = df["caption"].map(lambda text: extract_matches(text, PEOPLE_PATTERN))
    df["scene_matches"] = df["caption"].map(lambda text: extract_matches(text, SCENE_PATTERN))
    df["negative_matches"] = df["caption"].map(lambda text: extract_matches(text, NEGATIVE_PATTERN))

    df["has_people"] = df["people_matches"].map(bool)
    df["has_scene"] = df["scene_matches"].map(bool)
    df["has_negative"] = df["negative_matches"].map(bool)
    df["duration_ok"] = df["seconds"].between(args.min_seconds, args.max_seconds, inclusive="both")
    df["quality_score"] = compute_quality_score(df)

    base_mask = df["duration_ok"] & ~df["has_negative"]
    people_mask = base_mask & df["has_people"]
    scene_mask = base_mask & df["has_scene"] & ~df["has_people"]

    people_df = df.loc[people_mask].sort_values(
        by=["quality_score", "aesthetic score", "seconds"],
        ascending=[False, False, True],
        na_position="last",
    )
    scene_df = df.loc[scene_mask].sort_values(
        by=["quality_score", "aesthetic score", "seconds"],
        ascending=[False, False, True],
        na_position="last",
    )

    people_df = finalize_bucket(people_df, "people", args.max_people)
    scene_df = finalize_bucket(scene_df, "generic_scene", args.max_scene)

    selected_columns = [
        "bucket",
        "video",
        "caption",
        "seconds",
        "fps",
        "frame",
        "camera motion",
        "aesthetic score",
        "motion score",
        "temporal consistency score",
        "quality_score",
        "people_matches",
        "scene_matches",
        "negative_matches",
    ]
    selected_columns = [column for column in selected_columns if column in people_df.columns or column in scene_df.columns]

    output_df = pd.concat([people_df[selected_columns], scene_df[selected_columns]], ignore_index=True)
    output_df.to_csv(output_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"People bucket rows: {len(people_df)}")
    print(f"Generic scene bucket rows: {len(scene_df)}")
    print(f"Output rows: {len(output_df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()