class ContinualLearningTrainer:
    def __init__(self, continual_learner, params, state):
        self.continual_learner = continual_learner
        self.params = params
        self.state = state
