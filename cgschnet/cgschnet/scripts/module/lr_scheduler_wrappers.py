import torch

class SchedulerWrapper():
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    def __init__(self):
        self.scheduler = None

    def initialize(self, optimizer):
        """Instantiate the scheduler object"""
        raise NotImplementedError()

    def load_state_dict(self, state):
        assert self.scheduler is not None
        return self.scheduler.load_state_dict(state)

    def state_dict(self):
        assert self.scheduler is not None
        return self.scheduler.state_dict()

    def is_annealing(self):
        """Returns True if this is an annealing scheduler (one with a cyclic learning rate)"""
        return False

    def step_batch(self, fractional_epoch):
        """Called after every batch, fractional_epoch should be (epoch + batch/num_batches)."""
        pass

    def step(self, val_loss):
        """Called at the end of an epoch, after training and validation are finished"""
        pass

class SchedulerWrapper_ExponentialLR(SchedulerWrapper):
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    def __init__(self, gamma):
        self.gamma = gamma
        self.scheduler = None

    def initialize(self, optimizer):
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

    def step(self, val_loss):
        assert self.scheduler is not None
        self.scheduler.step()

    def __repr__(self):
        return f"SchedulerWrapper_ExponentialLR({self.gamma})"

class SchedulerWrapper_CosineAnnealingWarmRestarts(SchedulerWrapper):
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    def __init__(self, T_0, T_mult):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.scheduler = None

    def initialize(self, optimizer):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult)

    def is_annealing(self): #pyright: ignore[reportIncompatibleMethodOverride]
        return True

    def step_batch(self, fractional_epoch):
        assert self.scheduler is not None
        self.scheduler.step(fractional_epoch)

    def __repr__(self):
        return f"SchedulerWrapper_CosineAnnealingWarmRestarts({self.T_0}, {self.T_mult})"

class SchedulerWrapper_CosineAnnealingLR(SchedulerWrapper):
    def __init__(self, T_max, eta_min):
        self.T_max = T_max
        self.eta_min = eta_min
        self.scheduler = None

    def initialize(self, optimizer):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)

    def is_annealing(self): #pyright: ignore[reportIncompatibleMethodOverride]
        return True

    def step(self, val_loss):
        assert self.scheduler is not None
        self.scheduler.step()

    def __repr__(self):
        return f"SchedulerWrapper_CosineAnnealingLR({self.T_max}, {self.eta_min})"

class SchedulerWrapper_ReduceLROnPlateau(SchedulerWrapper):
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    def __init__(self, factor, patience, threshold, min_lr):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.threshold_mode = "abs"
        self.scheduler = None

    def initialize(self, optimizer):
        assert self.threshold_mode == "abs" or self.threshold_mode == "rel"
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.factor,
                                                                    patience=self.patience, min_lr=self.min_lr,
                                                                    threshold=self.threshold, threshold_mode=self.threshold_mode)

    def step(self, val_loss):
        assert self.scheduler is not None
        self.scheduler.step(val_loss)

    def __repr__(self):
        return f"SchedulerWrapper_ReduceLROnPlateau({self.factor}, {self.patience}, {self.threshold}, {self.min_lr})"
