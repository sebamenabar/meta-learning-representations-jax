class BaseTrainer:
    def __init__(
        self, forward, params, state, opt_state=None,
    ):
        self.__forward = forward
        self.params = params
        self.state = state
        self.opt_state = opt_state

        self.__step_fn = lambda x: x
