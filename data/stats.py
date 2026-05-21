import pandas as pd
import numpy as np
import re

# 1. Load data
df = pd.read_csv('OpenVidHD.csv')

# 2. Column names
print(f"Columns: {df.columns.tolist()}")

# 3. Numeric stats
cols_to_stat = {
    'seconds': 'seconds',
    'fps': 'fps',
    'frame': 'frame',
    'aesthetic score': 'aesthetic score',
    'motion score': 'motion score'
}

stats_res = {}
for name, col in cols_to_stat.items():
    if col in df.columns:
        stats_res[name] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75)
        }

print("\nNumeric Statistics:")
for k, v in stats_res.items():
    print(f"{k}: mean={v['mean']:.2f}, median={v['median']:.2f}, Q1={v['q25']:.2f}, Q3={v['q75']:.2f}")

# 4. Keyword filter
people_positive = r"person|people|man|woman|boy|girl|child|children|adult|male|female|human|face|hand|hands|portrait|couple|worker|chef|farmer|driver|dancer|skier|surfer|cyclist|runner|athlete|bride|groom"
generic_scene_positive = r"street|road|city|town|village|building|room|kitchen|bedroom|office|park|garden|forest|mountain|beach|sea|ocean|river|lake|sky|cloud|sunset|snow|field|farm|restaurant|cafe|shop|market|living room|table|car|bus|train|boat|bridge|stadium"
negative = r"animation|anime|cartoon|illustration|cgi|3d render|rendered|game footage|gameplay|logo|text overlay|watermark|slide|presentation|meme|poster|screenshot"

def check_match(text, pattern):
    if not isinstance(text, str): return False
    return bool(re.search(pattern, text, re.IGNORECASE))

df['match_people'] = df['caption'].apply(lambda x: check_match(x, people_positive))
df['match_scene'] = df['caption'].apply(lambda x: check_match(x, generic_scene_positive))
df['match_negative'] = df['caption'].apply(lambda x: check_match(x, negative))

count_people = df['match_people'].sum()
count_scene = df['match_scene'].sum()
count_negative = df['match_negative'].sum()
count_filtered = ((df['match_people'] | df['match_scene']) & ~df['match_negative']).sum()

print("\nCaption Filtering:")
print(f"People Positive: {count_people}")
print(f"Scene Positive: {count_scene}")
print(f"Negative: {count_negative}")
print(f"People/Scene and NOT Negative: {count_filtered}")

# 5. Estimation
total_size_tb = 4.5
total_videos = 433509
avg_orig_size_mb = (total_size_tb * 1024 * 1024) / total_videos
avg_seconds = df['seconds'].mean()

def calc_estimated_size(bitrate_mbps, seconds):
    # bitrate in Mbps, duration in seconds. Size in MB.
    # size = (bitrate * 10^6 * seconds) / 8 / 10^6 = bitrate * seconds / 8
    return (bitrate_mbps * seconds) / 8

size_15 = calc_estimated_size(1.5, avg_seconds)
size_25 = calc_estimated_size(2.5, avg_seconds)

print("\nEstimations:")
print(f"Average original video size: {avg_orig_size_mb:.2f} MB")
print(f"At 1.5 Mbps, avg size: {size_15:.2f} MB")
print(f"At 2.5 Mbps, avg size: {size_25:.2f} MB")
print(f"Total capacity for 10k videos (1.5 / 2.5 Mbps): {size_15*10000/1024:.2f} GB / {size_25*10000/1024:.2f} GB")
print(f"Total capacity for 50k videos (1.5 / 2.5 Mbps): {size_15*50000/1024:.2f} GB / {size_25*50000/1024:.2f} GB")

