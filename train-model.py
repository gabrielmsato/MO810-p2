import argparse
from pathlib import Path
from typing import Callable, Tuple
import dask.array as da
import dask.dataframe as df
import numpy as np
import time

from dasf_seismic.attributes.complex_trace import Envelope, InstantaneousFrequency, CosineInstantaneousPhase
from dasf.ml.xgboost import XGBRegressor
from dasf.transforms import ArraysToDataFrame, PersistDaskData, Transform
from dasf.pipeline import Pipeline
from dasf.datasets import Dataset
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.decorators import task_handler
from dasf.utils.types import is_dask_array

ENVELOPE = "ENVELOPE"
INST_FREQ = "INST-FREQ"
COS_INST_PHASE = "COS-INST-PHASE"

# get features
# pipeline get features
# relatorio
# pergunta do classroom
# load model 
# https://mljar.com/blog/xgboost-save-load-python/

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
            # features = da.append(features, n1, axis=0)
            features[str(n_features)]= n1
            n_features += 1


            # Padding for left border data
            n2[:,:,0] = 0
            # Rolling array by 1 position for left neighbor
            n2 = da.roll(data, -1, axis=2)
            # Adding neighbors in features array
            # features = da.append(features, n2, axis=0)
            features[str(n_features)]= n2

            n_features += 1


        # Get elements for trace window
        # Get bot neighbor
        n2 = data.copy()
        for i in range(self.trace_window):
            # Padding for bottom border data
            n1[:,700,:] = 0
            # Rolling array by 1 position for top neighbor
            n1 = da.roll(data, 1, axis=1)
            # Adding neighbors in features array
            # features = da.append(features, n1, axis=0)
            features[str(n_features)]= n1
            n_features += 1


            # Padding for top border data
            n2[:,0,:] = 0
            # Rolling array by 1 position for bot neighbor
            n2 = da.roll(data, -1, axis=1)
            # Adding neighbors in features array
            # features = da.append(features, n2, axis=0)
            features[str(n_features)]= n2
            n_features += 1


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
            features[str(n_features)]= n1
            n_features += 1

            # Padding for back inline dimension border data
            n2[0,:,:] = 0
            # Rolling array by 1 position for back inline dimension neighbor
            n2 = da.roll(data, -1, axis=0)
            # Adding neighbors in features array
            features[str(n_features)]= n2
            n_features += 1

        # print(n_features, data[:,0,0].size, data[0,:,0].size, data[0,0,:].size)
        # print(features.size)
        return features
    
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
    
class GetSelectedFeatures(Transform):
    """Class for get features for seismic attributes calculation
    """
    def __init__(self, axis: int, side:int, neighbor:int):
        """Extract features for seismic attributes calculation

        Parameters
        ----------
        axis: int
            Dimension in array
        side: int
            left= 0 or right= 1
        neighbor: int
            Which neighbor is desired

        Returns
        -------
        DaskArray
            Array with features
        """
        self.axis = axis
        self.side = side
        self.neighbor = neighbor

    def __transform_generic(self, data):
        """Extract selected features for seismic attributes calculation

        Parameters
        ----------
        data: DaskArray
            Dataset for feature extraction

        Returns
        -------
        DaskArray
            Array with features
        """
        if is_dask_array(data):
            feature = data.copy()
        else:
            feature = np.copy(data)

        # Sample window
        if (self.axis == 2):
            d_size = feature[0,0,:].size
            if self.side == 0:
                # Padding for left border data
                for i in range(self.neighbor):
                    feature[:,:,d_size-1] = feature[:,:,0]
                    # Rolling array by 1 position for left neighbor
                    if is_dask_array:
                        feature = da.roll(data, 1, axis=2)
                    else:
                        feature = np.roll(data, 1, axis=2)

            elif self.side == 1:
                # Padding for right border data
                for i in range(self.neighbor):
                    feature[:,:,0] = feature[:,:,d_size-1]
                    # Rolling array by 1 position for left neighbor
                    if is_dask_array:
                        feature = da.roll(data, -1, axis=2)
                    else:
                        feature = np.roll(data, -1, axis=2)
            else:
                raise Exception("Invalid Side")
            
        # Trace window
        elif (self.axis == 1):
            d_size = feature[0,:,0].size
            if self.side == 0:
                # Padding for left border data
                for i in range(self.neighbor):
                    feature[:,d_size-1,:] = feature[:,0,:]
                    # Rolling array by 1 position for top neighbor
                    if is_dask_array:
                        feature = da.roll(data, 1, axis=1)
                    else:
                        feature = np.roll(data, 1, axis=1)

            elif self.side == 1:
                # Padding for right border data
                for i in range(self.neighbor):
                    feature[:,0,:] = feature[:,d_size-1,:]
                    # Rolling array by 1 position for bot neighbor
                    if is_dask_array:
                        feature = da.roll(data, -1, axis=1)
                    else:
                        feature = np.roll(data, -1, axis=1)

            else:
                raise Exception("Invalid Side")
            
        # Inline window
        elif (self.axis == 0):
            d_size = feature[:,0,0].size
            if self.side == 0:
                # Padding for left border data
                for i in range(self.neighbor):
                    feature[d_size-1,:,:] = feature[0,:,:]
                    # Rolling array by 1 position for front inline dimension neighbor
                    if is_dask_array:
                        feature = da.roll(data, 1, axis=0)
                    else:
                        feature = np.roll(data, 1, axis=0)

            elif self.side == 1:
                # Padding for right border data
                for i in range(self.neighbor):
                    feature[0,:,:] = feature[d_size-1,:,:]
                    # Rolling array by 1 position for back inline dimension neighbor
                    if is_dask_array:
                        feature = da.roll(data, -1, axis=0)
                    else:
                        feature = np.roll(data, -1, axis=0)

            else:
                raise Exception("Invalid Side")
        else:
            raise Exception("Invalid dimension")
        
        
        return feature

    def _lazy_transform_cpu(self, data):
        return self.__transform_generic(data)
    
    def _transform_cpu(self, data):
        return self.__transform_generic(data)

    
    
class GetFeatures_fromDataframe(Transform):
    """Split dataframe get features
    """ 
    def __transform_generic(self, dataframe):
        """Get n - 1 first columns of dataframe

        Parameters
        ----------
        dataframe: Dask Dataframe
            Dataframe to get labels
        """
        return dataframe.iloc[:,:-1]

    def _lazy_transform_cpu(self, dataframe):
        return self.__transform_generic(dataframe)
    
    def _transform_cpu(self, dataframe):
        return self.__transform_generic(dataframe)
    
class GetLabels_fromDataframe(Transform):
    """Split dataframe to get labels
    """
    def __transform_generic(self, dataframe):
        """Get last column of dataframe

        Parameters
        ----------
        dataframe: Dask Dataframe
            Dataframe to get labelss
        """
        return dataframe["lbl"]
    
    def _lazy_transform_cpu(self, dataframe):
        return self.__transform_generic(dataframe)
    
    def transform(self, dataframe):
        return self.__transform_generic(dataframe)

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
        chunks = {0: "auto",1: -1, 2: -1} 
        self.chunks = chunks
        
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
                    attribute: str, 
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
    attribute: str
        Atributo sismico a ser calculado
    samples_window: int
        Numero de vizinhos na dimensao das amostras
    trace_window: int
        Numero de vizinhos na dimensao dos tracos 
    inline_window: int
        Numero de vizinhos na dimensao das inlines
    pipeline_save_location:
        Local onde a figura do pipeline criado será salva
    output:
        Local onde o modelo será salvo (.json)

    Returns
    -------
    Tuple[Pipeline, Callable]
        Uma tupla, onde o primeiro elemento é o pipeline e o segundo é último operador (kmeans.fit_predict), 
        de onde os resultados serão obtidos.
    """
    
    print("Criando pipeline....")
    # Declarando os operadores necessários
    dataset = MyDataset(name="F3 dataset", data_path=dataset_path)

    # Identifica o atributo escolhido
    if attribute == ENVELOPE:
        att = Envelope()
    elif attribute == INST_FREQ:
        att = InstantaneousFrequency()
    elif attribute == COS_INST_PHASE:
        att = CosineInstantaneousPhase()
    else:
        raise Exception("Invalid Attribute")

    array2df = ArraysToDataFrame()

    xgboost = XGBRegressor()

    features = GetFeatures_fromDataframe()
    labels = GetLabels_fromDataframe()
    
    # Compondo o pipeline
    pipeline = Pipeline(
        name="F3 seismic attributes",
        executor=executor
    )
    pipeline.add(dataset)

    dict = {"f0": dataset}
    for i in range(samples_window):
        feature_extractor = GetSelectedFeatures(axis=2, side=0, neighbor=i+1)
        pipeline.add(feature_extractor, data=dataset)
        dict["fsw_l_"+str(i+1)] = feature_extractor

        feature_extractor = GetSelectedFeatures(axis=2, side=1, neighbor=i+1)
        pipeline.add(feature_extractor, data=dataset)
        dict["fsw_r_"+str(i+1)] = feature_extractor 
    
    for i in range(trace_window):
        feature_extractor = GetSelectedFeatures(axis=1, side=0, neighbor=i+1)
        pipeline.add(feature_extractor, data=dataset)
        dict["ftw_l_"+str(i+1)] = feature_extractor

        feature_extractor = GetSelectedFeatures(axis=1, side=1, neighbor=i+1)
        pipeline.add(feature_extractor, data=dataset)
        dict["ftw_r_"+str(i+1)] = feature_extractor 
    
    for i in range(inline_window):
        feature_extractor = GetSelectedFeatures(axis=0, side=0, neighbor=i+1)
        pipeline.add(feature_extractor, data=dataset)
        dict["fiw_l_"+str(i+1)] = feature_extractor

        feature_extractor = GetSelectedFeatures(axis=0, side=1, neighbor=i+1)
        pipeline.add(feature_extractor, data=dataset)
        dict["fiw_r_"+str(i+1)] = feature_extractor 
    

    pipeline.add(att, X=dataset)
    dict["lbl"] = att

    print(dict.keys())

    pipeline.add(array2df, **dict)
    pipeline.add(features, dataframe=array2df)
    pipeline.add(labels, dataframe=array2df)

    # pipeline.add(persist, X=features2df)
    # pipeline.add(featurePersist, X=features2df)

    pipeline.add(xgboost.fit, X=features, y=labels)
    # pipeline.add(xgboost., fname=output)
    
    try:
        if pipeline_save_location is not None:
            pipeline.visualize(filename=pipeline_save_location)
    except Exception as e:
        print("Erro ao salvar imagem de pipeline")
    
    # Retorna o pipeline e o operador xgboost, donde os resultados serão obtidos
    return pipeline, xgboost.fit
    # return pipeline, persist

def run(pipeline: Pipeline, last_node: Callable) -> np.ndarray:
# def run(pipeline: Pipeline) -> np.ndarray:
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
    end = time.time()
    
    print(f"Feito! Tempo de execução: {end - start:.2f} s")
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")
    parser.add_argument("--attribute", type=str, required=True, help="Nome do atributo a ser usado para treinar o modelo.")
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
    #Salvando o modelo
    res.save_model(args.output)
    # res.to_csv("features_dataframe.csv")
    # print(f"O resultado é um array com o shape: {res.shape}")
