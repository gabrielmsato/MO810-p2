docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 train-model3.py --attribute "COS-INST-PHASE" --data data/F3_train3.zarr --samples-window 4 --trace-window 4 --inline-window 4 --address tcp://143.106.16.213:8786  --output "cos444_2.json"
