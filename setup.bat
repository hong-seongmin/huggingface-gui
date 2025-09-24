@echo off
setlocal EnableDelayedExpansion

REM HuggingFace GUI Auto Setup Script for Windows
REM Requires Windows 10+ and PowerShell

title HuggingFace GUI Setup

REM Configuration
set PYTHON_MIN_VERSION=3.9
set REQUIRED_MEMORY_GB=4
set SCRIPT_DIR=%~dp0

REM Colors (using PowerShell Write-Host)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo ==================================================
echo 🚀 HuggingFace GUI Auto Setup (Windows)
echo ==================================================
echo.

REM Check if PowerShell is available
powershell -Command "exit 0" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PowerShell is required but not available.
    echo Please install PowerShell 5.0 or later.
    pause
    exit /b 1
)

REM Functions using PowerShell for colored output
:log_info
powershell -Command "Write-Host '[INFO] %~1' -ForegroundColor Blue"
exit /b

:log_success
powershell -Command "Write-Host '[SUCCESS] %~1' -ForegroundColor Green"
exit /b

:log_warning
powershell -Command "Write-Host '[WARNING] %~1' -ForegroundColor Yellow"
exit /b

:log_error
powershell -Command "Write-Host '[ERROR] %~1' -ForegroundColor Red"
exit /b

REM Check Python installation
call :log_info "Checking Python installation..."

python --version >nul 2>&1
if errorlevel 1 (
    python3 --version >nul 2>&1
    if errorlevel 1 (
        call :log_error "Python is not installed or not in PATH."
        call :log_error "Please install Python %PYTHON_MIN_VERSION%+ from https://python.org"
        echo.
        echo Press any key to open Python download page...
        pause >nul
        start https://www.python.org/downloads/
        exit /b 1
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

REM Get Python version
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>nul') do set PYTHON_VERSION=%%i

REM Simple version check (comparing major.minor)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% LSS 3 (
    call :log_error "Python version %PYTHON_VERSION% is too old. Minimum required: %PYTHON_MIN_VERSION%"
    exit /b 1
)

if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 9 (
        call :log_error "Python version %PYTHON_VERSION% is too old. Minimum required: %PYTHON_MIN_VERSION%"
        exit /b 1
    )
)

call :log_success "Python %PYTHON_VERSION% is compatible"

REM Check system memory
call :log_info "Checking system memory..."

for /f "skip=1 tokens=2" %%i in ('wmic computersystem get TotalPhysicalMemory') do (
    if not "%%i"=="" (
        set /a MEMORY_GB=%%i/1024/1024/1024
        goto :memory_checked
    )
)
:memory_checked

if %MEMORY_GB% LSS %REQUIRED_MEMORY_GB% (
    call :log_warning "System has %MEMORY_GB%GB RAM. Minimum %REQUIRED_MEMORY_GB%GB recommended."
    call :log_warning "Application may run slowly with large models."
) else (
    call :log_success "System memory: %MEMORY_GB%GB (sufficient)"
)

REM Check GPU availability
call :log_info "Checking GPU availability..."

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    call :log_success "NVIDIA GPU detected"
    set GPU_AVAILABLE=1
) else (
    call :log_warning "No NVIDIA GPU detected. Using CPU only."
    set GPU_AVAILABLE=0
)

REM Check if Git is installed
call :log_info "Checking Git installation..."
git --version >nul 2>&1
if errorlevel 1 (
    call :log_warning "Git is not installed. Some features may not work."
    call :log_info "You can download Git from: https://git-scm.com/download/win"
) else (
    call :log_success "Git is available"
)

REM Install/check pip
call :log_info "Checking pip installation..."
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    call :log_info "Installing pip..."
    %PYTHON_CMD% -m ensurepip --upgrade
    if errorlevel 1 (
        call :log_error "Failed to install pip"
        pause
        exit /b 1
    )
) else (
    call :log_success "pip is available"
)

REM Upgrade pip
call :log_info "Upgrading pip..."
%PYTHON_CMD% -m pip install --upgrade pip >nul 2>&1
if not errorlevel 1 (
    call :log_success "pip upgraded successfully"
)

REM Setup environment
call :log_info "Setting up environment..."

REM Create .env file if it doesn't exist
if not exist "%SCRIPT_DIR%.env" (
    if exist "%SCRIPT_DIR%.env.example" (
        copy "%SCRIPT_DIR%.env.example" "%SCRIPT_DIR%.env" >nul
        call :log_success "Created .env file from template"
    ) else (
        call :log_warning ".env.example not found, creating basic .env file"
        (
            echo # HuggingFace GUI Configuration
            echo HOST=127.0.0.1
            echo PORT=8501
            echo FASTAPI_HOST=127.0.0.1
            echo FASTAPI_PORT=8000
            echo HF_MODEL_CACHE_DIR=%TEMP%\hf_model_cache
            echo DEFAULT_DEVICE=auto
            echo MAX_CONCURRENT_MODELS=2
            echo LOG_LEVEL=INFO
            echo STREAMLIT_SERVER_HEADLESS=true
            echo STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
        ) > "%SCRIPT_DIR%.env"
        call :log_success "Created basic .env file"
    )
) else (
    call :log_success ".env file already exists"
)

REM Create necessary directories
set CACHE_DIR=%TEMP%\hf_model_cache
if not exist "%CACHE_DIR%" (
    mkdir "%CACHE_DIR%" 2>nul
    call :log_success "Created model cache directory: %CACHE_DIR%"
)

set LOGS_DIR=%SCRIPT_DIR%logs
if not exist "%LOGS_DIR%" (
    mkdir "%LOGS_DIR%" 2>nul
    call :log_success "Created logs directory"
)

REM Install Python dependencies
call :log_info "Installing Python dependencies..."

cd /d "%SCRIPT_DIR%"

if exist "requirements.txt" (
    call :log_info "Installing from requirements.txt..."
    %PYTHON_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        call :log_error "Failed to install dependencies"
        pause
        exit /b 1
    )
    call :log_success "Dependencies installed successfully"
) else (
    call :log_error "requirements.txt not found"
    pause
    exit /b 1
)

REM Test installation
call :log_info "Testing installation..."

%PYTHON_CMD% -c "import streamlit, transformers, torch; print('All modules imported successfully')" >nul 2>&1
if errorlevel 1 (
    call :log_error "Installation test failed"
    call :log_error "Some required packages may not be installed correctly"
    pause
    exit /b 1
) else (
    call :log_success "Installation test passed"
)

echo.
echo ==================================================
call :log_success "🎉 Setup completed successfully!"
echo ==================================================
echo.
echo 📋 Next steps:
echo    1. Review/edit .env file if needed
echo    2. Run the application:
echo.
echo       # Streamlit version:
echo       %PYTHON_CMD% -m streamlit run app.py
echo.
echo       # Desktop version:
echo       %PYTHON_CMD% run.py
echo.
echo    3. Open http://localhost:8501 in your browser
echo.
echo 💡 Keep this window open for future reference
echo.

REM Optional: Ask if user wants to start the application immediately
echo Would you like to start the Streamlit application now? (y/N)
set /p START_NOW="> "

if /i "!START_NOW!"=="y" (
    call :log_info "Starting HuggingFace GUI..."
    %PYTHON_CMD% -m streamlit run app.py
) else (
    echo.
    echo You can start the application later using:
    echo %PYTHON_CMD% -m streamlit run app.py
    echo.
    pause
)

endlocal