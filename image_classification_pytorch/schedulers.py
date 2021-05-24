class ExponentialLR():
    def __init__(self,
                 gamma=0.999,
                 last_epoch=-1,
                 verbose=False,
                 interval='step'):
        super().__init__()

        self.gamma = gamma
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.interval = interval
