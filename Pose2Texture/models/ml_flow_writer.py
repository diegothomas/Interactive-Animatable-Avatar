import numpy as np
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME,MLFLOW_USER,MLFLOW_SOURCE_NAME
from omegaconf import DictConfig, ListConfig


class MlflowWriter():
    def __init__(self, experiment_name):
        self.client = MlflowClient()
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except Exception as e:
            print(e)
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.experiment = self.client.get_experiment(self.experiment_id)
        print("New experiment started")
        print(f"Name: {self.experiment.name}")
        print(f"Experiment_id: {self.experiment.experiment_id}")
        print(f"Artifact Location: {self.experiment.artifact_location}")

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                if parent_name != "model.aware":                                    #for Avatar In the Shell Proj
                    self.client.log_param(self.run_id, f'{parent_name}.{i}', v)
        else:
            self.client.log_param(self.run_id, f'{parent_name}', element)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_metric_step(self, key, value, step):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path) 
    
    def log_artifacts(self, local_path ):
        self.client.log_artifact(self.run_id, local_path )
    

    def log_dict(self, dictionary, file):
        self.client.log_dict(self.run_id, dictionary, file)
    
    def log_figure(self, figure, file):
        self.client.log_figure(self.run_id, figure, file)
        
    def set_terminated(self):
        self.client.set_terminated(self.run_id)

    def create_new_run(self, tags=None):
        self.run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = self.run.info.run_id
        print(f"New run started: {tags['mlflow.runName']}")

if __name__ == "__main__":
    #mlflow.set_tracking_uri("file://" + cwd + "/mlruns") #/mlrunsディレクトリの場所変えたかったらこれでできるよ
    EXPERIMENT_NAME = "hogeeeeeehogeeeee"
    writer = MlflowWriter(EXPERIMENT_NAME)

    for seed in [46,47,2021,0,1]: #朧げながら浮かんできたよ
        #タグにrunの情報等を入れられるよ
        #System tagsについて : https://mlflow.org/docs/latest/tracking.html#id19
        tags = {'trial':seed,
                MLFLOW_RUN_NAME:"runの名前を決められるよ",
                MLFLOW_USER:"ユーザーも決められるよ",
                MLFLOW_SOURCE_NAME:"ソースも決められるよ",
            }

        writer.create_new_run(tags) #新しいrunを作るよ

        # 〜〜〜〜 学習とかテストとかのコードは省略 ~~~~~~~~

        writer.log_param("learning_rate", 0.01)          #args => key and value
        writer.log_metric("test accuracy", 88)           #args => key and value
        writer.log_metric_step("train loss", 10, step=1) #args => key and value and step
        writer.log_metric_step("train loss", 5, step=2)  #args => key and value and step
        writer.log_metric_step("train loss", 2, step=3)  #args => key and value and step

        text = "Hi, I am Matsutakk"
        with open("text.txt", 'w') as f:
            f.write(text)
        writer.log_artifact("text.txt")   #　他にもpickleファイルの保存とかもできるよ

        dic = {'hoge':"hahaha"}
        writer.log_dict(dic, 'dic.json')  #　jsonとかyamlにもできるよ

        fig = plt.figure()
        x = np.linspace(0, 2*np.pi, 500)
        plt.plot(x,np.sin(x))
        writer.log_figure(fig, 'sin.png') #　図の保存用メソッド, 
        plt.close(fig)

        writer.set_terminated()           #必ず呼ぶこと！　ファイルのクローズみたいなもんだよ，きっと


    # その他メソッドを知りたい・追加したいなら以下を参考に
    # https://mlflow.org/docs/latest/python_api/mlflow.tracking.html