set MEX="c:\Program Files\MATLAB\R2011b\bin\mex.bat"
set MEXEXT=mexw32
set INSTALLDIR="c:\home\taras\mfann_test"
::start /w %MEX% fann_train_call.c
%MEX% fann_train_call.c .\x64\Release\fann_train_main.obj .\x64\Release\fann_my_io.obj .\lib\fannfloat.lib "C:\Program Files\MATLAB\R2011b\extern\lib\win64\microsoft\libut.lib"
::start /w xcopy /q /c fann_train_call.mexw32 %INSTALLDIR%
