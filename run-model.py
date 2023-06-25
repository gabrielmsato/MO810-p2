import argparse
from pathlib import Path
from typing import Callable, Tuple
import dask.array as da
import numpy as np
import time
import json

from dasf_seismic.attributes.complex_trace import Envelope, InstantaneousFrequency, CosineInstantaneousPhase
# from dasf.ml.xgboost import XGBRegressor
from dasf.transforms import ArraysToDataFrame, PersistDaskData, Transform
from dasf.pipeline import Pipeline
from dasf.datasets import Dataset
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.decorators import task_handler
from dasf.utils.types import is_dask_array

import xgboost as xgb

from dasf.transforms import Fit
from dasf.transforms import Predict
from dasf.transforms import FitPredict

ENVELOPE = "ENVELOPE"
INST_FREQ = "INST-FREQ"
COS_INST_PHASE = "COS-INST-PHASE"

class MyXGBRegressor(Predict):
    def __init__(
        self,
        max_depth=None,
        max_leaves=None,
        max_bin=None,
        grow_policy=None,
        learning_rate=None,
        n_estimators=100,
        verbosity=None,
        objective=None,
        booster=None,
        tree_method=None,
        n_jobs=None,
        gamma=None,
        min_child_weight=None,
        max_delta_step=None,
        subsample=None,
        sampling_method=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        reg_alpha=None,
        reg_lambda=None,
        scale_pos_weight=None,
        base_score=None,
        random_state=None,
        num_parallel_tree=None,
        monotone_constraints=None,
        interaction_constraints=None,
        importance_type=None,
        gpu_id=None,
        validate_parameters=None,
        predictor=None,
        enable_categorical=False,
        max_cat_to_onehot=None,
        eval_metric=None,
        early_stopping_rounds=None,
        callbacks=None,
        **kwargs
    ):

        self.__xgb_mcpu = xgb.dask.DaskXGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaves=max_leaves,
            max_bin=max_bin,
            grow_policy=grow_policy,
            learning_rate=learning_rate,
            verbosity=verbosity,
            objective=objective,
            booster=booster,
            tree_method=tree_method,
            n_jobs=n_jobs,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            sampling_method=sampling_method,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            random_state=random_state,
        )
        return self.__xgb_mcpu.load_model(kwargs["model"])

    
    def _lazy_predict_cpu(self, X, sample_weight=None, **kwargs):
        return self.__xgb_mcpu.predict(X=X, **kwargs)
    
    def _predict_cpu(self, X, sample_weight=None, **kwargs):
        return self.__xgb_cpu.predict(X=X, **kwargs)

    
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

# class GetFeatures_fromDataframe(Transform):
#     """Split dataframe get features
#     """ 
#     def __transform_generic(self, dataframe):
#         """Get n - 1 first columns of dataframe

#         Parameters
#         ----------
#         dataframe: Dask Dataframe
#             Dataframe to get labels
#         """
#         return dataframe.iloc[:,:-1]

#     def _lazy_transform_cpu(self, dataframe):
#         return self.__transform_generic(dataframe)
    
#     def _transform_cpu(self, dataframe):
#         return self.__transform_generic(dataframe)

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
        chunks = {0: "auto",1: -1, 2: -1} 
        
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
    
    array2df = ArraysToDataFrame()
    # features = GetFeatures_fromDataframe()

    xgboost = MyXGBRegressor(model=ml_model)
    
    # Compondo o pipeline
    pipeline = Pipeline(
        name="F3 seismic attributes",
        executor=executor
    )
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

    pipeline.add(array2df, **dict)
    pipeline.add(xgboost.predict, X=array2df)
    
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
                                            executor, args.ml_model, 
                                            args.samples_window, 
                                            args.trace_window, 
                                            args.inline_window, 
                                            pipeline_save_location="train_model_pipeline.svg")
    
    # Executamos e pegamos o resultado
    res = run(pipeline, last_node) 

    print(f"O resultado é um array com o shape: {res.shape}")
    # Salvando o atributo sismico
    np.save(args.output, res)

    #     # Podemos fazer o reshape e printar a primeira inline
    # res = res.values.reshape((401, 701, 255))
    # import matplotlib.pyplot as plt
    # plt.imsave("inline0.png", res[0], cmap="Reds")
    # print(f"Figura da inline 0 salva")
