@echo off
setlocal enabledelayedexpansion

timeout /t 2 >nul

REM Get the market index from the command line
set "MARKET_INDEX=%1"

REM Validate the market index
if /I "%MARKET_INDEX%"=="dow" (
    echo Using DOW index
    set "PORT=8100" 
) else if /I "%MARKET_INDEX%"=="ftse100" (
    echo Using FTSE 100 index
    set "PORT=8200"
) else if /I "%MARKET_INDEX%"=="sse50" (
    echo Using SSE 50 index
    set "PORT=8300"
) else (
    echo Invalid market index: %MARKET_INDEX%
    echo Must be one of: dow, ftse100, sse50
    exit /b 1
)

REM Define config filename and volume name
set "CONFIG_FILE=temp_config_%MARKET_INDEX%.yaml"
set "VOLUME_NAME=mydata"

REM Use Python to inject the tag into the YAML
python -c "import yaml, os; idx = os.environ['MARKET_INDEX']; cfg = yaml.safe_load(open('configs/config.yaml')); cfg['active_index'] = idx; yaml.dump(cfg, open(f'temp_config_{idx}.yaml', 'w'))"

REM Check if base image already exists
docker image inspect rl-base:py312-cu118-v1 >nul 2>&1
IF ERRORLEVEL 1 (
    echo Base image not found. Building base image...
    docker build -f Dockerfile.base -t rl-base:py312-cu118-v1 .
) ELSE (
    echo Base image already exists. Skipping build.
)

REM Build hyperparameter image with unique tag
echo Building hyperparameter image...
docker build -f Dockerfile.hyperparameters -t hyperparameter-image-%MARKET_INDEX% .

REM Create unique shared volume
docker volume inspect %VOLUME_NAME% >nul 2>&1
IF ERRORLEVEL 1 (
    echo Creating volume %VOLUME_NAME%...
    docker volume create %VOLUME_NAME%
) ELSE (
    echo Volume %VOLUME_NAME% already exists. Skipping creation.
)

REM Run hyperparameter container
echo Running hyperparameter container...
docker run ^
    --name hyperparameter-%MARKET_INDEX% ^
    -v "%cd%\%CONFIG_FILE%:/app/data/configs/config.yaml" ^
    -v %VOLUME_NAME%:/app/data ^
    -p %PORT%:8080 ^
    --env-file .env ^
    --gpus all ^
    --rm -it hyperparameter-image-%MARKET_INDEX%

REM Build reward tests image with unique tag
echo Building reward tests image...
docker build -f Dockerfile.rewardTests -t rewardtests-image-%MARKET_INDEX% .

REM Run reward tests container
echo Hyperparameter container finished. Starting reward tests...
docker run ^
    --name rewardtests-%MARKET_INDEX% ^
    -v %VOLUME_NAME%:/app/data ^
    -p %PORT%:8080 ^
    --env-file .env ^
    --gpus all ^
    --rm -it rewardtests-image-%MARKET_INDEX%

echo All containers completed successfully.

REM Clean up config file
del "%CONFIG_FILE%"