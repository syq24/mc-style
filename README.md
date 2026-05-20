# mc-style
beginning

cd /root/autodl-tmp/mc-style/data

wget -O OpenVidHD.csv "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv"

cd /root/autodl-tmp/mc-style/data

python3 filter_openvidhd_buckets.py \
  --max-people 18000 \
  --max-scene 12000 \
  --output /root/autodl-tmp/mc-style/data/OpenVidHD_30k_people_scene.csv

分片汇总表：data/OpenVidHD_30k_people_scene_parts_summary.csv

每条视频对应哪个分片：data/OpenVidHD_30k_people_scene_parts_mapped.csv

需要下载的 zip 清单：data/OpenVidHD_30k_people_scene_parts_zip_list.txt

缺失映射列表：data/OpenVidHD_30k_people_scene_parts_missing.txt
