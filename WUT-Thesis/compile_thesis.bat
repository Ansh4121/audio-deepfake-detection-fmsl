@echo off
echo Compiling LaTeX thesis...

echo Step 1: First compilation...
pdflatex main.tex

echo Step 2: Processing bibliography...
biber main

echo Step 3: Second compilation...
pdflatex main.tex

echo Step 4: Final compilation...
pdflatex main.tex

echo.
echo Compilation complete!
echo Check for main.pdf in the current directory.

pause
