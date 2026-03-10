class Attack:
    def __init__(
        self,
    ): ...

    def _resolve_local_mode(self, local: bool, tampering: float) -> bool:
        """
        Resolve whether to use local mode based on initialization setting and call parameters.

        Arguments:
            local: The local parameter passed to __call__
            tampering: The tampering percentage (0.0 to 1.0)

        Returns:
            True if local mode should be used, False for global mode.
            If None, use global only for tampering == 1
        """
        if local is not None:
            return local
        else:
            return False if tampering < 0.99 else True

    def __call__(
        self,
        text: str,
        tampering: float,
        local: bool,
    ) -> str:
        raise NotImplementedError


class NullAttack(Attack):
    def __init__(self):
        super().__init__()

    def __call__(self, text: str, tampering: float, local: bool) -> str:
        return text
