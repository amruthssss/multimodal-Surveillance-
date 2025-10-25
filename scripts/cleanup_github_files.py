"""
Clean up GitHub repository - keep only essential files
Removes duplicate docs, test files, and unnecessary scripts
"""

import os
from pathlib import Path

def cleanup_github_repo():
    print("="*70)
    print("🧹 CLEANING UP GITHUB REPOSITORY")
    print("="*70)
    
    github_dir = Path('../muli_modal_github')
    
    if not github_dir.exists():
        print("❌ GitHub folder not found")
        return
    
    # Files to DELETE (keep only essentials)
    files_to_delete = [
        # Duplicate/old documentation
        '1.5_DAY_ACTION_PLAN.md',
        '2DAY_FAST_TRACK.md',
        '2_DAY_EMERGENCY_PLAN.md',
        'ADD_MORE_PATTERNS_QUICKSTART.md',
        'AGENT_ACCURACY_REPORT.md',
        'ALL_EVENTS_DATASETS.md',
        'API_ACCURACY_ANALYSIS.md',
        'CLEANUP_UNNECESSARY_FILES.ps1',
        'COMPARISON_RESULTS.md',
        'COMPLETE_SYSTEM_SUMMARY.md',
        'DOWNLOAD_CORRECT_FILE.md',
        'DOWNLOAD_STATUS.md',
        'FINAL_RECOMMENDATION.md',
        'FIX_EMPTY_KNOWLEDGE.md',
        'FREE_AI_APIS_GUIDE.md',
        'GEMINI_API_ANALYSIS.md',
        'KAGGLE_CORRUPTION_FIX.md',
        'KAGGLE_DIAGNOSTIC_CELL.py',
        'KAGGLE_FIXES_APPLIED.md',
        'KAGGLE_LEARN_FROM_VIDEOS.py',
        'KAGGLE_NOTEBOOK.py',
        'KAGGLE_PATH_FIX.md',
        'KAGGLE_QUICK_START.md',
        'KAGGLE_TRAINING_GUIDE.md',
        'PATTERN_EXPANSION_GUIDE.md',
        'PATTERN_LEARNING_GUIDE.md',
        'PRESENTATION_GUIDE.md',
        'SOLUTION_SUMMARY.md',
        'START_HERE.md',
        'TRAINING_CHECKLIST.md',
        'ULTRA_FUSION_INTEGRATION.md',
        
        # Test/diagnostic scripts
        'analyze_video_content.py',
        'check_integration.py',
        'cleanup_workspace.py',
        'create_corrected_video.py',
        'create_optimized_video.py',
        'create_output_video.py',
        'diagnose_detection_issue.py',
        'diagnose_yolo_detections.py',
        'fine_tune_patterns.py',
        'kaggle_yolo_training.py',
        'learn_agent_web.py',
        'learn_from_kaggle_videos.py',
        'optimize_hybrid_weights.py',
        'organize_patterns_for_project.py',
        'quick_check.py',
        'quick_test_expansion.py',
        'quick_video_test.py',
        'quickstart_patterns.py',
        'setup_huggingface.py',
        'test_powerful_agent.py',
        'test_pure_agent_detailed.py',
        'test_smart_fusion.py',
        'test_ultra_powerful_agent.py',
        'test_web_learning_agent.py',
        'test_web_powered_agent.py',
        'test_with_video_output.py',
        'test_with_visual_output.py',
        'verify_kaggle_knowledge.py',
        'verify_yolo_model.py',
        
        # Duplicate/old processing scripts
        'process_enhanced_agent.py',
        'process_enhanced_hybrid.py',
        'process_fixed_hybrid.py',
        'process_hybrid_agent.py',
        'process_pure_agent.py',
        'process_smart_hybrid.py',
        
        # Duplicate web learning scripts
        'expand_patterns_from_web.py',
        'merge_patterns.py',
        
        # Unused utils
        'utils/pure_agent_detector.py',
        'utils/self_learning_agent.py',
        'utils/smart_fusion_engine.py',
        'utils/ultra_powerful_agent.py',
        'utils/web_image_learning.py',
        'utils/web_learning_agent.py',
        'utils/web_search_learning.py',
        
        # Scripts folder cleanup
        'scripts/run_pure_agent.py',
        'scripts/estimate_training_time.py',
        'scripts/auto_download_and_prepare.py',
        
        # Old model files
        'models/args.yaml',
        'models/class_mapping.py',
        'models/dataset.yaml',
        'models/learning_stats.json',
        'models/working_emotion_model_info.txt',
    ]
    
    deleted_count = 0
    
    print("\n🗑️  Deleting unnecessary files...")
    for file_path in files_to_delete:
        full_path = github_dir / file_path
        if full_path.exists():
            full_path.unlink()
            deleted_count += 1
            if deleted_count % 10 == 0:
                print(f"   Deleted {deleted_count} files...")
    
    print(f"\n✅ Deleted {deleted_count} unnecessary files")
    
    # Count remaining files
    remaining = len(list(github_dir.rglob('*')))
    print(f"✅ Remaining files: {remaining}")
    
    print("\n📋 ESSENTIAL FILES KEPT:")
    print("   ✅ Core detection scripts (app.py, hybrid_intelligent_agent.py, etc.)")
    print("   ✅ Main documentation (README, training guides)")
    print("   ✅ Frontend (React app)")
    print("   ✅ Utils (detection wrappers)")
    print("   ✅ Scripts (training, dataset prep)")
    print("   ✅ Config files (requirements.txt, package.json)")
    
    print("\n📤 COMMIT & PUSH CHANGES:")
    print("="*70)
    print("""
cd ../muli_modal_github
git add .
git commit -m "Clean up: Remove unnecessary docs and test files"
git push
    """)
    
    print("="*70)

if __name__ == "__main__":
    cleanup_github_repo()
