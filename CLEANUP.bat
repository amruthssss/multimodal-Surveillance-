@echo off
cls
echo ================================================
echo  PROJECT CLEANUP - Remove Unnecessary Files
echo ================================================
echo.
echo This will DELETE:
echo   - All .md documentation files (except README.md)
echo   - datasets/ folder (1.4+ GB)
echo   - Test video files
echo   - node_modules/ in root
echo   - yolo_dataset_final.zip
echo   - Old test scripts
echo   - Temporary output files
echo.
echo This will KEEP:
echo   - models/ folder (AI models)
echo   - surveillance-backend/ (Node.js backend)
echo   - surveillance-frontend/ (React frontend)
echo   - All Python detection files
echo   - app.py (Flask server)
echo   - requirements.txt
echo   - config/ folder
echo   - utils/ folder
echo   - README.md
echo.
echo ================================================
echo.
set /p confirm="Are you sure you want to proceed? (YES/no): "
if /i not "%confirm%"=="YES" (
    echo Cleanup cancelled.
    pause
    exit /b
)

echo.
echo [1/8] Removing documentation files...
del /q "AGENT_ACCURACY_REPORT.md" 2>nul
del /q "ALL_EVENTS_DATASETS.md" 2>nul
del /q "AUDIO_INTEGRATION_README.md" 2>nul
del /q "BACKEND_FRONTEND_INTEGRATION.md" 2>nul
del /q "CLEAN_PROJECT_SUMMARY.md" 2>nul
del /q "COMPARISON_RESULTS.md" 2>nul
del /q "DOWNLOAD_CORRECT_FILE.md" 2>nul
del /q "DOWNLOAD_STATUS.md" 2>nul
del /q "FINAL_RECOMMENDATION.md" 2>nul
del /q "FIX_EMPTY_KNOWLEDGE.md" 2>nul
del /q "FRONTEND_INTEGRATED.md" 2>nul
del /q "IMPLEMENTATION_COMPLETE.md" 2>nul
del /q "INTEGRATED_FEATURES.md" 2>nul
del /q "MODEL_READY.md" 2>nul
del /q "PATTERN_TO_YOLO_GUIDE.md" 2>nul
del /q "PROJECT_STRUCTURE.md" 2>nul
del /q "SURVEILLANCE_SETUP_GUIDE.md" 2>nul
del /q "TRAINING_GUIDE_RTX4060.md" 2>nul
del /q "TRAINING_README.md" 2>nul
del /q "ULTRA_FUSION_INTEGRATION.md" 2>nul
del /q "WHAT_TO_DO_NEXT.md" 2>nul
del /q "YOLO_OBJECT_DETECTION_80_20.md" 2>nul
echo Done!

echo [2/8] Removing datasets folder (1.4+ GB)...
rd /s /q "datasets" 2>nul
echo Done!

echo [3/8] Removing test video files...
del /q "hybrid_output_*.mp4" 2>nul
del /q "optimized_*.mp4" 2>nul
del /q "ultra_*.mp4" 2>nul
echo Done!

echo [4/8] Removing large dataset zip...
del /q "yolo_dataset_final.zip" 2>nul
echo Done!

echo [5/8] Removing root node_modules...
rd /s /q "node_modules" 2>nul
echo Done!

echo [6/8] Removing test scripts...
del /q "test_clean_white_text.py" 2>nul
del /q "test_dtest.py" 2>nul
del /q "test_enhanced_red_overlay.py" 2>nul
del /q "test_multiple_videos.py" 2>nul
del /q "test_report.json" 2>nul
del /q "test_road_accident.py" 2>nul
del /q "test_video.py" 2>nul
del /q "test_videoplayback.py" 2>nul
echo Done!

echo [7/8] Removing temporary files...
del /q "enhanced_features_log.csv" 2>nul
del /q "cleanup_project.ps1" 2>nul
del /q "check_reference_video.py" 2>nul
del /q "find_original_video.py" 2>nul
del /q "verify_dataset.py" 2>nul
echo Done!

echo [8/8] Removing old/redundant Python files...
del /q "hybrid_yolo_agent.py" 2>nul
del /q "final_ultra_system.py" 2>nul
del /q "optimized_hybrid_system.py" 2>nul
del /q "ultra_hybrid_system.py" 2>nul
del /q "create_optimized_dataset.py" 2>nul
del /q "learn_patterns.py" 2>nul
del /q "tune_accuracy.py" 2>nul
echo Done!

echo.
echo ================================================
echo  CLEANUP COMPLETE!
echo ================================================
echo.
echo Removed approximately 2+ GB of unnecessary files
echo.
echo Your project now contains only:
echo   - surveillance-backend/     (Node.js + MongoDB)
echo   - surveillance-frontend/    (React App)
echo   - models/                   (AI Models)
echo   - enhanced_final_ultra_system.py (Main Detection)
echo   - app.py                    (Flask Server)
echo   - train_yolo_from_patterns.py
echo   - Other essential files
echo.
pause
