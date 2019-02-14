#Devcloud Horovod Setup / Benchmark Run
env_name="hvd"
#environment variables

mkdir ~/lib
cd ~/lib
ln -s /glob/supplementary-software/versions/glibc/glibc_2_28/lib/libm.so.6 
export LD_LIBRARY_PATH=~/lib:$LD_LIBRARY_PATH

export CC=/glob/development-tools/versions/gcc-7.3.0/bin/gcc 
export LD_LIBRARY_PATH=/glob/development-tools/versions/gcc-7.3.0/lib64/:$LD_LIBRARY_PATH 
export PATH=/glob/development-tools/versions/gcc-7.3.0/bin/:$PATH 

export LD_LIBRARY_PATH=/glob/development-tools/mklml/lib/:${LD_LIBRARY_PATH}
source /glob/development-tools/parallel-studio/impi/2018.3.222/bin64/mpivars.sh 

#install conda
function install_env() {
	if [ `conda --version | grep "command not found" | wc -l` -eq 1 ]
	then
		cd ~/
		if ![ -f Anaconda3-5.3.0-Linux-x86_64.sh ]
		then
		wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh
		fi
		chmod +x Anaconda3-5.3.0-Linux-x86_64.sh
		rm -rf ~/anaconda3/
		./Anaconda3-5.3.0-Linux-x86_64.sh -b
	else
		echo "Anaconda previously installed."
	fi
	if ! [ `conda env list | grep $env_name | wc -l` -eq 1 ]
	then
		echo "creating environment '$env_name'.."
		conda create -n $env_name -c intel python=3.6 tensorflow-mkl=1.10 absl-py -y
	else
		source activate $env_name
	fi
	if [ `conda list | grep "tensorflow-mkl" | wc -l` -lt 1 ]
	then
		conda install tensorflow-mkl
	fi
	CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" pip install --upgrade --user --no-cache-dir horovod 
}
function run_benchmark() {
	#check for benchmarking repo
	cd ~/
	if [ -f ~/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py ]
	then
		echo TensorFlow benchmarks previously installed.
	else
		git clone https://www.github.com/tensorflow/benchmarks/
	fi
	cd benchmarks
	git checkout cnn_tf_v1.10_compatible

	source activate $env_name

	#get number of physical cores
	cores_per_socket=`lscpu | grep "Core(s)" | sed "s/^.*\([0-9]\{2\}[0-9]*\)$/\1/"`
	sockets=`lscpu | grep "Socket(s)" | sed "s/^.*\([0-9]\)$/\1/"`
	num_cores=$(($cores_per_socket * $sockets))

	#environment variables (mpirun)
	export inter_op=2
	export intra_op=$num_cores
	export MODEL=resnet50
	export python_script=~/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

	mpirun python $python_script --mkl=True --model resnet50 --batch_size 64 --data_format NCHW --num_intra_threads $intra_op --num_inter_threads 2 --kmp_blocktime=0 --distortions=False --num_batches=30 --variable_update=horovod --horovod_device=cpu 
}
if [ `echo $HOSTNAME | grep "n" | wc -l` -lt 1 ]
then
	echo "Please run in an interactive job (call 'qsub -I')"
else	
	#menu
	while true; do
	read -p "Menu: [I]nstall environment [R]un benchmark [Q]uit	" input
	case $input in
		[Ii]* ) install_env; break;;
		[Rr]* ) run_benchmark; break;;
		[Qq]* ) break;;
	esac
	done
fi
