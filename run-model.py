import argparse
from pathlib import Path
from typing import Callable, Tuple
import dask.array as da
import numpy as np
import time

from dasf_seismic.attributes.complex_trace import Envelope, InstantaneousFrequency, CosineInstantaneousPhase
from dasf.ml.xgboost import XGBRegressor
from dasf.transforms import ArraysToDataFrame, PersistDaskData, Transform
from dasf.pipeline import Pipeline
from dasf.datasets import Dataset
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.decorators import task_handler

ENVELOPE = "ENVELOPE"
INST_FREQ = "INST-FREQ"
COS_INST_PHASE = "COS-INST-PHASE"

class GetFeatures(Transform):
    """Class for get features for seismic attributes calculation
    """
    def __init__(self, samples_window: int, trace_window: int, inline_window: int):
        """Extract features for seismic attributes calculation

        Parameters
        ----------
        sample_window: int
            Number of neighbors in sample dimension
        trace_window: int
            Number of neighbors in trace dimension
        inline_window: int
            Number of neighbors in inline dimension

        Returns
        -------
        DaskArray
            Array with features
        """
        self.samples_window = samples_window
        self.trace_window = trace_window
        self.inline_window = inline_window

    def _lazy_transform_cpu(self, data):
        """Extract features for seismic attributes calculation

        Parameters
        ----------
        data: DaskArray
            Dataset for feature extraction

        Returns
        -------
        DaskArray
            Array with features
        """
        features = data.copy()

        # Get elements for sample window

        n1 = data.copy()
        n2 = data.copy()
        n_features = 1
        for i in range(self.samples_window):
            # Padding for right border data
            n1[:,:,254] = 0
            # Rolling array by 1 position for right neighbor
            n1 = da.roll(data, 1, axis=2)
            # Adding neighbors in features array
            features = da.append(features, n1, axis=0)

            # Padding for left border data
            n2[:,:,0] = 0
            # Rolling array by 1 position for left neighbor
            n2 = da.roll(data, -1, axis=2)
            # Adding neighbors in features array
            features = da.append(features, n2, axis=0)

            n_features += 2


        # Get elements for trace window
        # Get bot neighbor
        n2 = data.copy()
        for i in range(self.trace_window):
            # Padding for bottom border data
            n1[:,700,:] = 0
            # Rolling array by 1 position for top neighbor
            n1 = da.roll(data, 1, axis=1)
            # Adding neighbors in features array
            features = da.append(features, n1, axis=0)

            # Padding for top border data
            n2[:,0,:] = 0
            # Rolling array by 1 position for bot neighbor
            n2 = da.roll(data, -1, axis=1)
            # Adding neighbors in features array
            features = da.append(features, n2, axis=0)

            n_features += 2


        # Get elements for inline window
        # Get top neighbor
        n1 = data.copy()
        # Get bot neighbor
        n2 = data.copy()
        for i in range(self.inline_window):
            # Padding for front inline dimension border data
            n1[9,:,:] = 0
            # n1[400,:,:] = 0
            # Rolling array by 1 position for front inline dimension neighbor
            n1 = da.roll(data, 1, axis=0)
            # Adding neighbors in features array
            features = da.append(features, n1, axis=0)

            # Padding for back inline dimension border data
            n2[0,:,:] = 0
            # Rolling array by 1 position for back inline dimension neighbor
            n2 = da.roll(data, -1, axis=0)
            # Adding neighbors in features array
            features = da.append(features, n2, axis=0)

            n_features += 2

        print(n_features, data[:,0,0].size, data[0,:,0].size, data[0,0,:].size)
        print(features.size)
        return da.reshape(features, (n_features, data[:,0,0].size, data[0,:,0].size, data[0,0,:].size))
    
    def _transform_cpu(self, data):
        """Extract features for seismic attributes calculation

        Parameters
        ----------
        data: DaskArray
            Dataset for feature extraction

        Returns
        -------
        DaskArray
            Array with features
        """
        f_size = data.size(data)
        features = np.copy(data).reshape((1, f_size))

        # Get elements for sample window
        # Get top neighbor
        n1 = np.copy(self.ata)
        # Get bot neighbor
        n2 = np.copy(data)
        for i in range(self.samples_window):
            # Padding for bottom data
            n1[:,:,254] = 0
            # Rolling array by 1 position for top neighbor
            n1 = np.roll(data, 1, axis=2)
            # Adding neighbors in features array
            features = np.append(features, np.reshape(n1, (1, f_size)), axis=0)

            # Padding for bottom data
            n2[:,:,0] = 0
            # Rolling array by 1 position for bot neighbor
            n2 = np.roll(data, -1, axis=2)
            # Adding neighbors in features array
            features = np.append(features, np.reshape(n2, (1, f_size)), axis=0)

        # Get elements for trace window
        # Get top neighbor
        n1 = np.copy(data)
        # Get bot neighbor
        n2 = np.copy(data)
        for i in range(self.trace_window):
            # Padding for bottom data
            n1[:,700,:] = 0
            # Rolling array by 1 position for top neighbor
            n1 = np.roll(data, 1, axis=1)
            # Adding neighbors in features array
            features = np.append(features, np.reshape(n1, (1, f_size)), axis=0)

            # Padding for bottom data
            n2[:,0,:] = 0
            # Rolling array by 1 position for bot neighbor
            n2 = np.roll(data, -1, axis=1)
            # Adding neighbors in features array
            features = np.append(features, np.reshape(n2, (1, f_size)), axis=0)

        # Get elements for inline window
        # Get top neighbor
        n1 = np.copy(data)
        # Get bot neighbor
        n2 = np.copy(data)
        for i in range(self.inline_window):
            # Padding for bottom data
            n1[400,:,:] = 0
            # Rolling array by 1 position for top neighbor
            n1 = np.roll(data, 1, axis=0)
            # Adding neighbors in features array
            features = np.append(features, np.reshape(n1, (1, f_size)), axis=0)

            # Padding for bottom data
            n2[0,:,:] = 0
            # Rolling array by 1 position for bot neighbor
            n2 = np.roll(data, -1, axis=0)
            # Adding neighbors in features array
            features = np.append(features, np.reshape(n2, (1, f_size)), axis=0)


        return features

class MyDataset(Dataset):
    """Classe para carregar dados de um arquivo .npy ou .zarr
    """
    def __init__(self, name: str, data_path: str, chunks: str = "32Mb"):
        """Instancia um objeto da classe MyDataset

        Parameters
        ----------
        name : str
            Nome simbolicamente associado ao dataset
        data_path : str
            Caminho para o arquivo .npy
        chunks: str
            Tamanho dos chunks para o dask.array
        """
        super().__init__(name=name)
        self.data_path = data_path
        self.chunks = chunks
        chunks = {0: "auto",1: "auto", 2: -1} 
        
    def _lazy_load_cpu(self):
        return da.from_zarr(self.data_path, chunks=self.chunks)
    
    def _load_cpu(self):
        return np.load(self.data_path)
    
    @task_handler
    def load(self):
        ...

def create_executor(address: str=None) -> DaskPipelineExecutor:
    """Cria um DASK executor

    Parameters
    ----------
    address : str, optional
        Endereço do Scheduler, by default None

    Returns
    -------
    DaskPipelineExecutor
        Um executor Dask
    """
    if address is not None:
        addr = ":".join(address.split(":")[:2])
        port = str(address.split(":")[-1])
        print(f"Criando executor. Endereço: {addr}, porta: {port}")
        return DaskPipelineExecutor(local=False, use_gpu=False, address=addr, port=port)
    else:
        return DaskPipelineExecutor(local=True, use_gpu=False)

def create_pipeline(dataset_path: str, 
                    executor: DaskPipelineExecutor, 
                    ml_model: str, 
                    samples_window: int, 
                    trace_window: int, 
                    inline_window: int, 
                    pipeline_save_location: str = None) -> Tuple[Pipeline, Callable]:
    """Cria o pipeline DASF para ser executado

    Parameters
    ----------
    dataset_path : str
        Caminho para o arquivo .zarr
    executor : DaskPipelineExecutor
        Executor Dask
    ml_model: str
        Modelo de machine learning (.json)
    samples_window: int
        Numero de vizinhos na dimensao das amostras
    trace_window: int
        Numero de vizinhos na dimensao dos tracos 
    inline_window: int
        Numero de vizinhos na dimensao das inlines
    pipeline_save_location:
        Local onde a figura do pipeline criado será salva

    Returns
    -------
    Tuple[Pipeline, Callable]
        Uma tupla, onde o primeiro elemento é o pipeline e o segundo é último operador (kmeans.fit_predict), 
        de onde os resultados serão obtidos.
    """
    
    print("Criando pipeline....")
    # Declarando os operadores necessários
    dataset = MyDataset(name="F3 dataset", data_path=dataset_path)

    # Extraindo as features do dado
    features = GetFeatures(samples_window=samples_window, trace_window=trace_window, inline_window=inline_window)
    
    features2df = ArraysToDataFrame()
    # Persist é super importante! Se não cada partial_fit do k-means vai computar o grafo até o momento!
    # Usando persist, garantimos que a computação até aqui já foi feita e está em memória distribuida.
    featurePersist = PersistDaskData()

    xgboost = XGBRegressor()
    
    xgboost.load_model(ml_model)
    
    # Compondo o pipeline
    pipeline = Pipeline(
        name="F3 seismic attributes",
        executor=executor
    )
    pipeline.add(dataset)

    pipeline.add(features, data=dataset)
    pipeline.add(features2df, **features)
    pipeline.add(featurePersist, X=features2df)


    pipeline.add(xgboost.load_model, fname=ml_model)


    pipeline.add(xgboost.predict, X=featurePersist)
    
    try:
        if pipeline_save_location is not None:
            pipeline.visualize(filename=pipeline_save_location)
    except Exception as e:
        print("Erro ao salvar imagem de pipeline")
    
    # Retorna o pipeline e o operador kmeans, donde os resultados serão obtidos
    return pipeline, xgboost.predict

def run(pipeline: Pipeline, last_node: Callable) -> np.ndarray:
    """Executa o pipeline e retorna o resultado

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline a ser executado
    last_node : Callable
        Último operador do pipeline, de onde os resultados serão obtidos

    Returns
    -------
    np.ndarray
        NumPy array com os resultados
    """
    print("Executando pipeline")
    start = time.time()
    pipeline.run()
    res = pipeline.get_result_from(last_node)
    res = res.compute()
    end = time.time()
    
    print(f"Feito! Tempo de execução: {end - start:.2f} s")
    return res

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
    pipeline, last_node = create_pipeline(args.data, 
                                            executor, args.attribute, 
                                            args.samples_window, 
                                            args.trace_window, 
                                            args.inline_window, 
                                            pipeline_save_location="train_model_pipeline.svg")
    
    # Executamos e pegamos o resultado
    res = run(pipeline, last_node) 

    print(f"O resultado é um array com o shape: {res.shape}")
    # Salvando o atributo sismico
    np.save(args.output, res)
