#!/usr/bin/env bash
# Run from inside the sim2field/ folder, logged in to GitHub.
set -e
git init
git add .
git commit -m "Sim2Field: scan-to-eval deployment check for robot policies"
git branch -M main
git remote add origin https://github.com/khurramscience/sim2field.git 2>/dev/null || git remote set-url origin https://github.com/khurramscience/sim2field.git
git push -u origin main
echo "Pushed. Now enable GitHub Pages (Settings -> Pages -> main / root)."
