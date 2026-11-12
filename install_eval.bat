@echo off
"C:\Users\mikecovlee\miniconda3\envs\llm-eval\python.exe" -m pip install "lm-eval[api,math,ifeval]==0.4.12" "math-verify[antlr4_11_0]" transformers nltk langdetect immutabledict sympy > "%~dp0\pipit.log" 2>&1
echo EXITCODE %ERRORLEVEL% >> "%~dp0\pipit.log"
echo ALLDONE >> "%~dp0\pipit.log"
