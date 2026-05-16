@echo off
title OmniVoice - Sync from upstream
echo ============================================
echo   Sync from upstream (k2-fsa/OmniVoice)
echo   then push to origin (thanhng8/omnivoice-tool)
echo ============================================
echo.

cd /d "%~dp0.."

REM ----------------------------------------------------------------
REM  Make sure the 'upstream' remote exists. First run will add it.
REM ----------------------------------------------------------------
git remote get-url upstream >nul 2>&1
if errorlevel 1 (
    echo [setup] 'upstream' remote not found - adding k2-fsa/OmniVoice ...
    git remote add upstream https://github.com/k2-fsa/OmniVoice.git
    if errorlevel 1 goto :error
    echo.
)

echo [1/3] Fetching upstream (k2-fsa/OmniVoice)...
git fetch upstream
if errorlevel 1 goto :error

echo.
echo [2/3] Merging upstream/master into local main...
git merge upstream/master --no-edit
if errorlevel 1 (
    echo.
    echo *** MERGE CONFLICT detected ***
    echo.
    echo The most likely conflict is README.md ^(rewritten in this fork^).
    echo Other files in tool/, main.png, voice_prompts/ should be safe.
    echo.
    echo Resolve manually, then run:
    echo   git add ^<files^>
    echo   git commit
    echo   git push origin main
    echo.
    echo Tip: to keep your README, run:
    echo   git checkout --ours README.md ^&^& git add README.md
    goto :end
)

echo.
echo [3/3] Pushing to origin (thanhng8/omnivoice-tool)...
git push origin main
if errorlevel 1 goto :error

echo.
echo ============================================
echo   Done! Repo synced successfully.
echo ============================================
goto :end

:error
echo.
echo *** ERROR: Command failed. Check output above. ***

:end
echo.
pause
