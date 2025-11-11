class Attack:

    def __init__(
        self,
    ):
        pass

    def __call__(
        self,
        text: str,
        tampering: float,
        local: bool,
    ) -> str:
        raise NotImplementedError

