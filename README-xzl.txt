README-xzl.txt

pi5 build

cmake .
cmake --build . --config Release

## model conversion
# default fp16
python3 python/convert_pytorch_to_ggml.py \
/data/models/pi-deployment/04b-tunefull-x58-562.pth \
/data/models/pi-deployment/04b-tunefull-x58-562.bin FP16


run

python3 python/generate_completions.py /data/models/pi-deployment/RWKV-5-World-0.1B-v1-20230803-ctx4096.bin
