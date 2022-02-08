class AIExtensionError(Exception):
    def __init__(self, where: str = "·") -> None:
        msg = (
            f"There has been an error processing this pipeline (ERROR - {where}).\n"
            "Please contact support via help@fluidattacks.com or open an issue in https://github.com/fluidattacks/ai-extension-docker"
        )
        super(AIExtensionError, self).__init__(msg)


class CommitRiskError(Exception):
    def __init__(self) -> None:
        msg = (
            "FluidAttacks AI system detected a mean risk in your commit files greater than the established limit\n"
            "(COMMIT_MEAN_RISK_LIMIT = 75%).\n"
        )
        super(CommitRiskError, self).__init__(msg)
