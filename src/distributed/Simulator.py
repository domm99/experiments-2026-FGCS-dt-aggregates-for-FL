import glob
import pandas as pd
from typing import Callable
from src.distributed.DT import DT
from src.distributed.DTAggregate import DTAggregate
from src.distributed.LearningConfig import LearningConfig

class Simulator:

    def __init__(self, data_folder: str, time: pd.Timestamp, config: LearningConfig, time_increase_strategy: Callable[[pd.Timestamp], pd.Timestamp], seed: int):
        self.time_increase_strategy = time_increase_strategy
        self.data_folder = data_folder
        self.seed = seed
        self.time = time
        self._config = config
        self._all_dts = self.init_dts()
        self.active_dts = self.get_active_dts()
        self._dt_aggregate = DTAggregate(config, seed)
        for dt in self._all_dts:
            dt.dt_aggregate = self._dt_aggregate

    def init_dts(self):
        files = glob.glob(f'{self.data_folder}/*.csv')
        dts = []
        for file in files:
            dt_id = file.split('/')[-1].split('.')[0]
            new_dt = DT(dt_id, self.data_folder, self._config)
            dts.append(new_dt)
        return dts

    def get_active_dts(self):
        return [dt for dt in self._all_dts if dt.is_active(self.time)]

    def start(self):
        pass