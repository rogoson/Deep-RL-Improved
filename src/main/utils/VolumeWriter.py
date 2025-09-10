import shutil
from pathlib import Path


def copyOver():
    """
    Copy files over to mountDir
    """
    sourceDir = Path(__file__).resolve().parent.parent.parent.parent / "src"
    configDir = Path(__file__).resolve().parent.parent.parent.parent / "configs"
    dataDir = Path(__file__).resolve().parent.parent.parent.parent / "data"

    dataDir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        sourceDir / "main" / "agents", dataDir / "agents", dirs_exist_ok=True
    )
    shutil.copytree(
        sourceDir / "main" / "animations", dataDir / "animations", dirs_exist_ok=True
    )
    shutil.copytree(
        sourceDir / "main" / "plotsAndPortfolioTrajectories",
        dataDir / "plotsAndPortfolioTrajectories",
        dirs_exist_ok=True,
    )

    shutil.copytree(configDir, dataDir / "configs", dirs_exist_ok=True)
