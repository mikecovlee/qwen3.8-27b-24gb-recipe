@echo off
echo START %DATE% %TIME% > "%~dp0\pipit3.log"
"C:\Users\mikecovlee\miniconda3\envs\llm-eval\python.exe" -u -m pip install "lm-eval[api,math,ifeval]==0.4.12" "math-verify[antlr4_11_0]" transformers nltk langdetect immutabledict sympy 1>> "%~dp0\pipit3.log" 2>&1
echo EXITCODE %ERRORLEVEL% >> "%~dp0\pipit3.log"
echo ALLDONE >> "%~dp0\pipit3.log"
