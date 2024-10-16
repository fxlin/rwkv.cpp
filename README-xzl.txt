README-xzl.txt

pi5 build

cmake .
cmake --build . --config Release -j4 --target rwkv

cmake --build . --config Debug -j4

cmake --build . --config Debug -j4

cmake --build . --config Debug -j4 --target rwkv

cmake --build .. --config Debug -j4 --target rwkv

## model conversion
# default fp16
python3 python/convert_pytorch_to_ggml.py \
/data/models/pi-deployment/04b-tunefull-x58-562.pth \
/data/models/pi-deployment/04b-tunefull-x58-562.bin FP16


## run
python3 python/generate_completions.py /data/models/pi-deployment/RWKV-5-World-0.1B-v1-20230803-ctx4096.bin

python3 python/generate_completions.py /data/models/pi-deployment/04b-tunefull-x58-562.bin


python3 python/inference_example.py /data/models/pi-deployment/04b-tunefull-x58-562.bin


measurement: 
RPI5

/usr/bin/time  -v python3 python/generate_completions.py /data/models/pi-deployment/04b-tunefull-x58-562.bin
System info: AVX=0 AVX2=0 AVX512=0 FMA=0 NEON=1 ARM_FMA=1 F16C=0 FP16_VA=0 WASM_SIMD=0 BLAS=0 SSE3=0 VSX=0
Loading RWKV model
2d tensor emb.weight: ne0 1024 x ne1 65536
2d tensor blocks.0.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.0.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.0.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.0.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.0.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.0.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.0.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.0.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.0.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.0.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.0.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.0.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.0.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.1.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.1.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.1.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.1.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.1.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.1.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.1.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.1.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.1.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.1.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.1.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.1.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.1.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.2.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.2.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.2.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.2.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.2.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.2.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.2.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.2.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.2.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.2.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.2.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.2.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.2.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.3.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.3.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.3.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.3.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.3.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.3.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.3.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.3.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.3.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.3.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.3.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.3.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.3.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.4.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.4.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.4.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.4.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.4.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.4.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.4.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.4.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.4.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.4.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.4.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.4.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.4.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.5.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.5.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.5.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.5.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.5.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.5.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.5.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.5.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.5.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.5.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.5.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.5.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.5.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.6.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.6.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.6.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.6.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.6.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.6.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.6.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.6.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.6.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.6.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.6.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.6.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.6.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.7.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.7.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.7.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.7.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.7.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.7.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.7.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.7.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.7.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.7.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.7.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.7.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.7.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.8.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.8.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.8.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.8.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.8.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.8.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.8.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.8.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.8.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.8.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.8.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.8.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.8.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.9.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.9.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.9.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.9.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.9.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.9.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.9.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.9.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.9.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.9.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.9.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.9.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.9.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.10.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.10.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.10.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.10.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.10.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.10.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.10.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.10.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.10.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.10.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.10.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.10.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.10.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.11.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.11.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.11.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.11.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.11.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.11.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.11.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.11.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.11.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.11.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.11.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.11.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.11.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.12.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.12.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.12.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.12.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.12.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.12.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.12.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.12.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.12.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.12.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.12.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.12.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.12.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.13.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.13.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.13.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.13.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.13.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.13.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.13.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.13.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.13.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.13.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.13.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.13.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.13.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.14.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.14.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.14.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.14.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.14.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.14.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.14.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.14.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.14.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.14.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.14.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.14.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.14.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.15.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.15.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.15.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.15.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.15.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.15.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.15.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.15.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.15.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.15.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.15.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.15.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.15.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.16.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.16.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.16.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.16.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.16.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.16.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.16.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.16.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.16.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.16.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.16.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.16.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.16.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.17.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.17.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.17.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.17.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.17.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.17.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.17.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.17.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.17.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.17.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.17.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.17.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.17.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.18.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.18.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.18.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.18.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.18.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.18.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.18.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.18.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.18.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.18.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.18.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.18.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.18.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.19.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.19.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.19.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.19.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.19.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.19.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.19.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.19.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.19.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.19.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.19.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.19.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.19.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.20.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.20.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.20.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.20.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.20.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.20.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.20.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.20.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.20.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.20.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.20.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.20.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.20.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.21.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.21.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.21.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.21.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.21.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.21.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.21.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.21.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.21.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.21.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.21.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.21.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.21.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.22.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.22.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.22.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.22.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.22.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.22.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.22.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.22.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.22.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.22.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.22.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.22.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.22.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor blocks.23.att.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.23.att.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.23.att.key1.weight: ne0 1024 x ne1 128
2d tensor blocks.23.att.key2.weight: ne0 128 x ne1 1024
2d tensor blocks.23.att.value1.weight: ne0 1024 x ne1 128
2d tensor blocks.23.att.value2.weight: ne0 128 x ne1 1024
2d tensor blocks.23.att.output.weight: ne0 1024 x ne1 1024
2d tensor blocks.23.att.gate1.weight: ne0 1024 x ne1 128
2d tensor blocks.23.att.gate2.weight: ne0 128 x ne1 1024
2d tensor blocks.23.ffn.key.weight: ne0 1024 x ne1 3584
2d tensor blocks.23.ffn.receptance1.weight: ne0 1024 x ne1 128
2d tensor blocks.23.ffn.receptance2.weight: ne0 128 x ne1 1024
2d tensor blocks.23.ffn.value.weight: ne0 3584 x ne1 1024
2d tensor head.weight: ne0 1024 x ne1 65536
Loading World v20230424 tokenizer
13 tokens in prompt

--- Generation 0 ---


Alice was so tired when she got back home so she went[ to bed early and slept for a while.
She got up at 4:30 and got ready for school. She had to be back at the school at 5:30.
She had to get dressed and go to school. She had to get ready for school and then she had to get home.
She had to get to school and get her clothes. She had to get home.
She had to get home and get her things. She had to get home.
She had]

Took 8.855 sec, 88 ms per token 11.29 toks/sec

--- Generation 1 ---


Alice was so tired when she got back home so she went[ to bed. She was tired and she was afraid. She got up at 6:30 and went to bed at 8:30. She was so tired and she was afrai
d. She got up at 8:30 and went to bed at 8:30.                                                                                                                                 
Alice was so tired and she was afraid. She got up at 6:30 and went to bed at 8:30. She was so tired and she was afraid. She got up at 6:30 and went to bed at 8]

Took 8.634 sec, 86 ms per token 11.58 toks/sec

--- Generation 2 ---


Alice was so tired when she got back home so she went[ to bed and fell asleep.
She woke up and found that her daughter had been taken away. She went to the
police station and told them what had happened. The police told her that she
had to report the incident to the police station and that they would take her
into custody.
She was taken to the police station and she was told that she had to report the
incident to the police station and that she would have to take her daughter
to the police station. She]

Took 8.624 sec, 86 ms per token 11.60 toks/sec
        Command being timed: "python3 python/generate_completions.py /data/models/pi-deployment/04b-tunefull-x58-562.bin"
        User time (seconds): 53.54
        System time (seconds): 1.79
        Percent of CPU this job got: 150%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:36.66
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 1251872     <<<<<<<<<<<<<< this 
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 95078
        Voluntary context switches: 2047
        Involuntary context switches: 19144
        Swaps: 0
        File system inputs: 867672
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 16384
        Exit status: 0
