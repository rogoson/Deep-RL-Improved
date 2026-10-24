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

    for source, destination in [
        (sourceDir / "main" / "agents", dataDir / "agents"),
        (sourceDir / "main" / "animations", dataDir / "animations"),
        (
            sourceDir / "main" / "plotsAndPortfolioTrajectories",
            dataDir / "plotsAndPortfolioTrajectories",
        ),
        (configDir, dataDir / "configs"),
    ]:
        if source.exists():  # may not exist yet if a run failed early
            shutil.copytree(source, destination, dirs_exist_ok=True)
