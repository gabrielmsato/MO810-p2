import argparse
from pathlib import Path
from typing import Callable, Tuple
import dask.array as da
import numpy as np
import time

from dasf_seismic.attributes.complex_trace import Envelope, InstantaneousFrequency, CosineInstantaneousPhase
from dasf.ml.xgboost import XGBoost
from dasf.transforms import ArraysToDataFrame, PersistDaskData
from dasf.pipeline import Pipeline
from dasf.datasets import Dataset
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.decorators import task_handler

ENVELOPE = "ENVELOPE"
INST_FREQ = "INST-FREQ"
COS_INST_PHASE = "COS-INST-PHASE"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")
    parser.add_argument("--ml-model", type=str, required=True, help="Nome do atributo a ser usado para treinar o modelo.")
    parser.add_argument("--data", type=str, required=True, help="Nome do arquivo com o dado sísmico de entrada .zarr")
    parser.add_argument("--samples-window", type=int, required=True, help="Número de vizinhos na dimensão das amostras de um traço.")
    parser.add_argument("--trace-window", type=int, required=True, help="Número de vizinhos na dimensão dos traços de uma inline.")
    parser.add_argument("--inline-window", type=int, default=None, help="Número de vizinhos na dimensão das inlines.")
    parser.add_argument("--address", type=str, default=None, help="Endereço do dask scheduler para execução do código.")
    parser.add_argument("--output", type=str, default=None, help="Nome do arquivo de saída onde será gravado o modelo treinado.")
    args = parser.parse_args()

    
   
    # Criamos o executor
    executor = create_executor(args.address)
    # Depois o pipeline
    pipeline, last_node = create_pipeline(args.data, executor, args.attribute, pipeline_save_location="train_model_pipeline.svg")
    # Executamos e pegamos o resultado
    res = run(pipeline, last_node)
    print(f"O resultado é um array com o shape: {res.shape}")
    