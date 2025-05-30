#!/bin/bash
# Git Identity Setup Template for IM2ELEVATION Project
# This script template shows how to set up Git identity for this specific project
# to maintain privacy across different projects and users

echo "🔧 Setting up Git identity for IM2ELEVATION project..."

# Check if we're in the right directory
if [[ ! -f "train.py" ]] || [[ ! -f "test.py" ]]; then
    echo "❌ Error: Please run this script from the IM2ELEVATION project root directory"
    exit 1
fi

# TODO: Replace with your actual GitHub information
GITHUB_USERNAME="your-github-username"
GITHUB_EMAIL="your-email@domain.com"

# Prompt for user information if not set
if [[ "$GITHUB_USERNAME" == "your-github-username" ]]; then
    echo "📝 Please enter your GitHub information:"
    read -p "GitHub Username: " GITHUB_USERNAME
    read -p "Email Address: " GITHUB_EMAIL
fi

# Set local Git configuration (project-specific only)
git config --local user.name "$GITHUB_USERNAME"
git config --local user.email "$GITHUB_EMAIL"

echo "✅ Git identity configured for this project:"
echo "   Username: $(git config --local user.name)"
echo "   Email: $(git config --local user.email)"
echo ""
echo "🔐 Privacy Note: This configuration is LOCAL to this project only."
echo "   Other projects and users won't inherit these settings."
echo ""
echo "💡 Tip: Copy this script to create your personal version:"
echo "   cp tools/git_setup/setup_git_identity_template.sh tools/git_setup/my_git_setup.sh"
echo "🎯 Ready for commits with proper attribution!"
