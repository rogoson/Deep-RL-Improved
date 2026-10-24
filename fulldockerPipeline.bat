@echo off
setlocal enabledelayedexpansion

timeout /t 2 >nul

REM Get the market index from the command line
set "MARKET_INDEX=%1"

REM Optional second arg "quick": minutes-long pipeline check (skips hyperparameter tuning)
set "QUICK_TEST=0"
if /I "%2"=="quick" set "QUICK_TEST=1"

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

REM Define volume name
set "VOLUME_NAME=mydata"
if "%QUICK_TEST%"=="1" (
    echo QUICK TEST MODE - results go to volume mydata_quicktest
    set "VOLUME_NAME=mydata_quicktest"
)

REM temp_config_<market>.yaml is generated inside the container from configs/config.yaml

REM Check if base image already exists
docker image inspect rl-base:py312-cu118-v1 >nul 2>&1
IF ERRORLEVEL 1 (
    echo Base image not found. Building base image...
    docker build -f Dockerfile.base -t rl-base:py312-cu118-v1 .
) ELSE (
    echo Base image already exists. Skipping build.
)

REM Create unique shared volume
docker volume inspect %VOLUME_NAME% >nul 2>&1
IF ERRORLEVEL 1 (
    echo Creating volume %VOLUME_NAME%...
    docker volume create %VOLUME_NAME%
) ELSE (
    echo Volume %VOLUME_NAME% already exists. Skipping creation.
)

if "%QUICK_TEST%"=="1" goto rewardtests

REM Build hyperparameter image with unique tag
echo Building hyperparameter image...
docker build -f Dockerfile.hyperparameters -t hyperparameter-image-%MARKET_INDEX% .

REM Run hyperparameter container
echo Running hyperparameter container...
docker run ^
    --name hyperparameter-%MARKET_INDEX% ^
    -v %VOLUME_NAME%:/app/data ^
    --env MARKET_INDEX=%MARKET_INDEX% ^
    -v marketdata:/app/marketDataCache ^
    --env MARKET_DATA_CACHE=/app/marketDataCache ^
    -p %PORT%:8080 ^
    --env-file .env ^
    --gpus all ^
    --rm -it hyperparameter-image-%MARKET_INDEX%
IF ERRORLEVEL 1 (
    echo Hyperparameter container FAILED - not starting reward tests.
    exit /b 1
)

:rewardtests
REM Build reward tests image with unique tag
echo Building reward tests image...
docker build -f Dockerfile.rewardTests -t rewardtests-image-%MARKET_INDEX% .

REM Run reward tests container
echo Hyperparameter container finished. Starting reward tests...
docker run ^
    --name rewardtests-%MARKET_INDEX% ^
    -v %VOLUME_NAME%:/app/data ^
    --env MARKET_INDEX=%MARKET_INDEX% ^
    -v marketdata:/app/marketDataCache ^
    --env MARKET_DATA_CACHE=/app/marketDataCache ^
    --env QUICK_TEST=%QUICK_TEST% ^
    -p %PORT%:8080 ^
    --env-file .env ^
    --gpus all ^
    --rm -it rewardtests-image-%MARKET_INDEX%
IF ERRORLEVEL 1 (
    echo Reward tests container FAILED - see traceback above.
    exit /b 1
)

echo All containers completed successfully.