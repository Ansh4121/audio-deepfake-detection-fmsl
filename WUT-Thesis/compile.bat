@echo off
echo Compiling LaTeX thesis...

set MIKTEX_PATH=C:\Users\anshc\AppData\Local\Programs\MiKTeX\miktex\bin\x64

echo Step 1: First compilation...
"%MIKTEX_PATH%\pdflatex.exe" -interaction=nonstopmode main.tex

echo Step 2: Processing bibliography...
"%MIKTEX_PATH%\biber.exe" main

echo Step 3: Second compilation...
"%MIKTEX_PATH%\pdflatex.exe" -interaction=nonstopmode main.tex

echo Step 4: Final compilation...
"%MIKTEX_PATH%\pdflatex.exe" -interaction=nonstopmode main.tex

echo.
if exist main.pdf (
    echo SUCCESS: PDF created successfully!
    echo File: main.pdf
    start main.pdf
) else (
    echo ERROR: PDF was not created!
)

pause
