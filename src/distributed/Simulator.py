import glob
import pandas as pd
from src.decentralized.DT import DT
from src.common import seed_everything

class Simulator:

    def __init__(self, data_folder: str, time: pd.Timestamp, time_increase_strategy: callable[int, int], seed: int):
        self.time_increase_strategy = time_increase_strategy
        self.data_folder = data_folder
        self.seed = seed
        self.time = time
        self.all_dts = self.init_dts()
        self.active_dts = self.get_active_dts()

        seed_everything(seed)


    def init_dts(self):
        files = glob.glob(f'{self.data_folder}/*.csv')
        dts = []
        for file in files:
            dt_id = file.split('/')[-1].split('.')[0]
            new_dt = DT(dt_id, self.data_folder)
            dts.append(new_dt)
        return dts

    def get_active_dts(self):
        return [dt for dt in self.all_dts if dt.is_active(self.time)]

    def start(self):
        pass