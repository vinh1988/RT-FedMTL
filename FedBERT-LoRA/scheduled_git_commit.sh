#!/bin/bash
# Scheduled Git Commit & Push Script
# Created: Thu Oct  9 07:42:23 AM +07 2025
# Scheduled for: 3 hours from creation

cd /home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA

echo '=== 🚀 SCHEDULED GIT COMMIT & PUSH ==='
echo 'Time: $(date)'
echo 'Branch: $(git branch --show-current)'
echo 'Checking for changes...'

# Check if there are any changes
if [ $(git status --porcelain | wc -l) -gt 0 ]; then
    echo '📦 Changes detected, proceeding with commit...'
    git status --short
    
    echo 'Adding all changes...'
    git add .
    
    echo 'Committing changes...'
    git commit -m "Scheduled commit: Auto-save experiment progress and results

- Automated commit at $(date)
- Includes any new experiment results  
- Preserves work progress
- Ready for analysis"
    
    echo 'Pushing to remote...'
    git push origin bert-fed-comprehensive-4scenario
    
    echo '✅ Scheduled commit completed successfully!'
    echo '📊 Summary:'
    echo '- Branch: $(git branch --show-current)'
    echo '- Latest commit: $(git log -1 --oneline)'
    echo '- Remote status: $(git status -sb)'
else
    echo 'ℹ️ No changes to commit - working tree clean'
    echo '📊 Current status:'
    git status -sb
fi

echo '=== ✅ SCHEDULED TASK COMPLETED ==='
