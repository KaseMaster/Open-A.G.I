#!/usr/bin/env python3
"""
Verification script for Market Pulse AGI project structure
"""

import os
import sys
from pathlib import Path

def verify_project_structure():
    """Verify that the project structure is correct"""
    
    # Define expected structure
    expected_dirs = [
        "contracts",
        "backend",
        "frontend",
        "agents",
        "docs",
        "tests",
        "scripts"
    ]
    
    expected_files = [
        "README.md",
        "contracts/MarketPulseWorkflow.sol",
        "backend/main.py",
        "backend/requirements.txt",
        "frontend/package.json",
        "agents/agent_base.py",
        "docs/project_analysis_and_roadmap.md",
        "docs/detailed_task_tracker.md",
        "docs/agent_orchestration_architecture.md",
        "docs/next_immediate_todos.md"
    ]
    
    project_root = Path(__file__).parent.parent
    print(f"Verifying project structure in: {project_root}")
    
    # Check directories
    print("\nChecking directories:")
    for directory in expected_dirs:
        dir_path = project_root / directory
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} (missing)")
            return False
    
    # Check files
    print("\nChecking files:")
    all_files_present = True
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_files_present = False
    
    return all_files_present

def verify_documentation():
    """Verify that documentation files have content"""
    project_root = Path(__file__).parent.parent
    
    doc_files = [
        "docs/project_analysis_and_roadmap.md",
        "docs/detailed_task_tracker.md",
        "docs/agent_orchestration_architecture.md",
        "docs/next_immediate_todos.md"
    ]
    
    print("\nChecking documentation files:")
    all_docs_valid = True
    for doc_file in doc_files:
        full_path = project_root / doc_file
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:  # Basic check for meaningful content
                        print(f"  ✓ {doc_file} (content verified)")
                    else:
                        print(f"  ? {doc_file} (file exists but may be empty)")
                        all_docs_valid = False
            except Exception as e:
                print(f"  ✗ {doc_file} (error reading file: {e})")
                all_docs_valid = False
        else:
            print(f"  ✗ {doc_file} (missing)")
            all_docs_valid = False
    
    return all_docs_valid

def main():
    """Main verification function"""
    print("Market Pulse AGI - Project Structure Verification")
    print("=" * 50)
    
    # Verify project structure
    structure_ok = verify_project_structure()
    
    # Verify documentation
    docs_ok = verify_documentation()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Project structure: {'PASS' if structure_ok else 'FAIL'}")
    print(f"  Documentation: {'PASS' if docs_ok else 'FAIL'}")
    
    if structure_ok and docs_ok:
        print("\n✓ All verification checks passed!")
        print("The Market Pulse AGI project structure is correctly set up.")
        return 0
    else:
        print("\n✗ Some verification checks failed.")
        print("Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())