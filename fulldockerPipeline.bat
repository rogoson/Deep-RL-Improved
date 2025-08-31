@echo off
setlocal enabledelayedexpansion

REM Get the arg from the command line
set "MARKET_INDEX=%1"

REM Validate the arg
if /I "%MARKET_INDEX%"=="dow" (
    echo Using DOW index
) else if /I "%MARKET_INDEX%"=="ftse100" (
    echo Using FTSE 100 index
) else if /I "%MARKET_INDEX%"=="sse50" (
    echo Using SSE 50 index
) else (
    echo Invalid market index: %MARKET_INDEX%
    echo Must be one of: dow, ftse100, sse50
    exit /b 1
)

set "TEMP_CONFIG=temp_config.yaml"

REM Use Python to inject the tag into the YAML
python -c "import yaml; cfg = yaml.safe_load(open('configs/config.yaml')); cfg['active_index'] = r'%MARKET_INDEX%'; yaml.dump(cfg, open(r'%TEMP_CONFIG%', 'w'))"

REM Check for optional flag to skip base image build
IF "%2"=="--skip-base" (
    echo Skipping base image build
) ELSE (
    echo Building base image...
    docker build -f Dockerfile.base -t rl-base:py312-cu118-v1 .
)

echo Building hyperparameter image...
docker build -f Dockerfile.hyperparameters -t hyperparameter-image .

echo Creating shared volume...
docker volume create mydata

echo Running hyperparameter container...
docker run ^
    -v mydata:/app/data ^
    -p 8080:8080 ^
    --env-file .env ^
    --gpus all ^
    --rm -it hyperparameter-image

echo Building reward tests image...
docker build -f Dockerfile.rewardTests -t rewardtests-image .

echo Hyperparameter container finished. Starting reward tests...
docker run ^
    -v mydata:/app/data ^
    -p 8090:8090 ^
    --env-file .env ^
    --gpus all ^
    --rm -it rewardtests-image

echo All containers completed successfully.

del "%TEMP_CONFIG%"