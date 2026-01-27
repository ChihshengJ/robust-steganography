class Attack:

    def __init__(
        self,
        local_mode: bool | None = None,
    ):
        """
        Initialize the attack.

        Arguments:
            local_mode: Controls local vs global attack behavior.
                - None (default): Use the legacy behavior where `local and tampering < 0.99`
                  determines whether to use local mode. At 100% tampering, global is used.
                - True: Force local mode (sentence-level) regardless of tampering level.
                  Even at 100% tampering, each sentence is processed individually.
                - False: Force global mode regardless of the `local` parameter passed to __call__.
        """
        self.local_mode = local_mode

    def _resolve_local_mode(self, local: bool, tampering: float) -> bool:
        """
        Resolve whether to use local mode based on initialization setting and call parameters.

        Arguments:
            local: The local parameter passed to __call__
            tampering: The tampering percentage (0.0 to 1.0)

        Returns:
            True if local mode should be used, False for global mode.
        """
        if self.local_mode is not None:
            # Explicit local_mode setting overrides everything
            return self.local_mode
        else:
            # Legacy behavior: use local only if local=True and tampering < 0.99
            return local and tampering < 0.99

    def __call__(
        self,
        text: str,
        tampering: float,
        local: bool,
    ) -> str:
        raise NotImplementedError
