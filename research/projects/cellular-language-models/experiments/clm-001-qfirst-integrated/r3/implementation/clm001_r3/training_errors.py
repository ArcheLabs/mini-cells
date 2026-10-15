from clm001_r2.training import NumericalFailure


class CreditFailure(NumericalFailure):
    def __init__(self, message):
        super().__init__("competition", {"message": message})
