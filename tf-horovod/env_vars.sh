cd ~/lib
ln -s /glob/supplementary-software/versions/glibc/glibc_2_28/lib/libm.so.6
export LD_LIBRARY_PATH=~/lib:$LD_LIBRARY_PATH
cd $cwd
export CC=/glob/development-tools/versions/gcc-7.3.0/bin/gcc 
export LD_LIBRARY_PATH=/glob/development-tools/versions/gcc-7.3.0/lib64/:$LD_LIBRARY_PATH 
export PATH=/glob/development-tools/versions/gcc-7.3.0/bin/:$PATH 