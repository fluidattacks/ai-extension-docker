class CredentialsError(Exception):
    def __init__(self) -> None:
        msg = (
            "There has been an error with your Azure credentials (username + token). They may not be correct.\n"
            "Please refer to https://github.com/fluidattacks/ai-extension-azuredevops#configuring-the-pipeline"
        )
        super(CredentialsError, self).__init__(msg)


class AIExtensionError(Exception):
    def __init__(self, where: str = "·") -> None:
        msg = (
            f"There has been an error processing this pipeline (ERROR - {where}).\n"
            "Please contact support via help@fluidattacks.com or open an issue in https://github.com/fluidattacks/ai-extension-azuredevops"
        )
        super(AIExtensionError, self).__init__(msg)


class CommitRiskError(Exception):
    def __init__(self) -> None:
        msg = (
            "FluidAttacks AI system detected a mean risk in your commit files greater than the established limit\n"
            "(COMMIT_MEAN_RISK_LIMIT = 80%).\n"
        )
        super(CommitRiskError, self).__init__(msg)
