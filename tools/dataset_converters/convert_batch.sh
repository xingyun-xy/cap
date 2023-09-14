
#转换车道线
# python3  /workspace/cap-sh/tools/dataset_converters/lane_parsing2horizon_format.py \
#     --data_dir /workspace/data/changan_data/lane_parsing/qianshi_duan/DTS000000933 \
#     --save_dir /workspace/data/changan_data/lane_parsing/qianshi_duan_horizon_format/horizon_format_933 \
#     --log_path /workspace/data/changan_data/lane_parsing/qianshi_duan/DTS000000933.txt

# python3  /workspace/cap-sh/tools/dataset_converters/lane_parsing2horizon_format.py \
#     --data_dir /workspace/data/changan_data/lane_parsing/zhoushi/DTS0000001144 \
#     --save_dir /workspace/data/changan_data/lane_parsing/zhoushi_horizon_format/horizon_format_1144 \
#     --log_path /workspace/data/changan_data/lane_parsing/zhoushi/DTS0000001144.txt

#转换全景分割
# python3  /workspace/cap-sh/tools/dataset_converters/semantic_parsing2horizon_format.py \
#     --data_dir /workspace/data/changan_data/semantic_parsing/qianshi_duan/DTS000000933 \
#     --save_dir /workspace/data/changan_data/semantic_parsing/qianshi_duan_horizon_format/horizon_format_933 \
#     --log_path /workspace/data/changan_data/semantic_parsing/qianshi_duan/DTS000000933.txt

#转换车头车尾
python3  /workspace/cap-sh/tools/dataset_converters/rear2horizon_format.py \
    --data_dir /workspace/data/changan_data/rear_detection/qianshi_chang/DTS000000962 \
    --save_dir /workspace/data/changan_data/rear_detection/qianshi_chang_horizon_format/horizon_format_962 \
    --log_path /workspace/data/changan_data/rear_detection/qianshi_chang/DTS000000962.txt

# python3  /workspace/cap-sh/tools/dataset_converters/rear2horizon_format.py \
#     --data_dir /workspace/data/changan_data/rear_detection/qianshi_chang/DTS000000924 \
#     --save_dir /workspace/data/changan_data/rear_detection/qianshi_chang_horizon_format/horizon_format_924 \
#     --log_path /workspace/data/changan_data/rear_detection/qianshi_chang/DTS000000924.txt

# python3  /workspace/cap-sh/tools/dataset_converters/rear2horizon_format.py \
#     --data_dir /workspace/data/changan_data/rear_detection/zhoushi/DTS000000927 \
#     --save_dir /workspace/data/changan_data/rear_detection/zhoushi_horizon_format/horizon_format_927 \
#     --log_path /workspace/data/changan_data/rear_detection/zhoushi/DTS000000927.txt

#转换骑车人+关键点
# python3  /workspace/cap-sh/tools/dataset_converters/cyclist2horizon_format.py \
#     --data_dir /workspace/data/changan_data/cyclist_keypoints/qianshi_duan/DTS000000949/ \
#     --save_dir /workspace/data/changan_data/cyclist_keypoints/qianshi_duan_horizon_format/DTS000000949-test/ \
#     --log_path /workspace/data/changan_data/cyclist_keypoints/qianshi_duan/DTS000000949-test.txt


#运行指令
#bash /workspace/cap-sh/tools/dataset_converters/convert_batch.sh