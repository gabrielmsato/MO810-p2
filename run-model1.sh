docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 run-model.py --ml-model cos000_3.json --data data/F3_train.zarr --samples-window 0 --trace-window 0 --inline-window 0 --address tcp://localhost:8786 --output data/cos000_2.npy
