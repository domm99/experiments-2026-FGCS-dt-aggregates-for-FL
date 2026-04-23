from src.distributed.DTAggregate import DTAggregate
from src.distributed.Simulator import Event, Simulator

class RetrainingPolicy:

    def on_event(self, event: Event, simulator: Simulator, dt_aggregate: DTAggregate):
        pass


## TODO implement policies
class PeriodicRetrainingPolicy(RetrainingPolicy):
    ...

class ActivationCountRetrainingPolicy(RetrainingPolicy):
    ...

class PerformanceDriftRetrainingPolicy(RetrainingPolicy):
    ...
