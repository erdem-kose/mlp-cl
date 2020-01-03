clc; clear;
curpath=mfilename('fullpath');
curfile=mfilename();
curpath=curpath(1:(end-size(curfile,2)));
cd(curpath);

mex_file='backpropagation_core_mex';
if exist([mex_file '.mexw64'],'file')~=0
    delete([mex_file '.mexw64']);
end
mex -v COPTIMFLAGS="-O2" backpropagation_core_mex.c CXXFLAGS="$CXXFLAGS -fopenacc -fopenmp" LDFLAGS="$LDFLAGS -fopenacc -fopenmp" [-Minfo=accel]

mex_file='choquetbackpropagation_core_mex';
if exist([mex_file '.mexw64'],'file')~=0
    delete([mex_file '.mexw64']);
end
mex -v COPTIMFLAGS="-O2" choquetbackpropagation_core_mex.c CXXFLAGS="$CXXFLAGS -fopenacc -fopenmp" LDFLAGS="$LDFLAGS -fopenacc -fopenmp" [-Minfo=accel]

mex_file='choquetnode_core_mex';
if exist([mex_file '.mexw64'],'file')~=0
    delete([mex_file '.mexw64']);
end
mex -v COPTIMFLAGS="-O2" choquetnode_core_mex.c CXXFLAGS="$CXXFLAGS -fopenacc -fopenmp" LDFLAGS="$LDFLAGS -fopenacc -fopenmp" [-Minfo=accel]

mex_file='multilayerperceptron_core_mex';
if exist([mex_file '.mexw64'],'file')~=0
    delete([mex_file '.mexw64']);
end
mex -v COPTIMFLAGS="-O2" multilayerperceptron_core_mex.c CXXFLAGS="$CXXFLAGS -fopenacc -fopenmp" LDFLAGS="$LDFLAGS -fopenacc -fopenmp" [-Minfo=accel]