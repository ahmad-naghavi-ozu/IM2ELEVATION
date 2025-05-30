# 🔧 Git Setup Tools

This directory contains scripts for setting up Git identity configuration for the IM2ELEVATION project in a privacy-conscious way.

## 📁 Files

### `setup_git_identity_template.sh` ✅ (Tracked)
**Public template** for setting up Git identity. Safe to track in version control.

- **Purpose**: Template showing how to configure Git identity for this project
- **Usage**: Copy and customize for personal use
- **Privacy**: Contains no personal information
- **Trackable**: ✅ Safe for public repositories

### `setup_git_identity.sh` ❌ (Ignored)
**Personal script** with hardcoded identity. Excluded from version control.

- **Purpose**: Quick setup script with personal GitHub information
- **Usage**: Run directly for instant Git identity configuration
- **Privacy**: Contains personal email and username
- **Trackable**: ❌ Excluded via `.gitignore` for privacy

## 🚀 Usage

### For Contributors (Using Template)
```bash
# Copy the template and customize it
cp tools/git_setup/setup_git_identity_template.sh tools/git_setup/my_git_setup.sh

# Edit with your information
nano tools/git_setup/my_git_setup.sh

# Run your customized script
chmod +x tools/git_setup/my_git_setup.sh
./tools/git_setup/my_git_setup.sh
```

### For Project Owner (Using Personal Script)
```bash
# Run the personal script (if available)
./tools/git_setup/setup_git_identity.sh
```

## 🔐 Privacy Design

This approach ensures:
- **Personal information** stays private (excluded from Git)
- **Template approach** is shared (tracked in Git)
- **Local-only identity** prevents cross-project contamination
- **Multi-user safety** on shared development servers

## 🎯 Best Practices

1. **Never commit personal scripts** with hardcoded credentials
2. **Always use local Git config** (`--local`) for project-specific identity
3. **Create personal copies** of templates for your own use
4. **Keep templates generic** and documentation clear

**Privacy-first Git identity management! 🛡️**
