#!/bin/bash
#SBATCH -N 1
#SBATCH -A m1759
#SBATCH -S 4
#SBATCH -J gppKerKokkos_KNL
#SBATCH -p debug
#SBATCH -t 00:30:00 
#SBATCH -C knl,quad,cache

##SBATCH -L SCRATCH
#
#module use /global/common/cori/software/modulefiles
#module unload advisor
#module load advisor/2018.integrated_roofline

export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#EXEC=gppKerFort.ex
#EXEC=gppKer.ex
EXEC=gppKer_gpuComplexOpenMP.ex
Input="512 2 32768 20"

num_mpi_ranks=1
numHT=4
threads_per_rank=$((64/${num_mpi_ranks}))
numLC=$(( ${threads_per_rank}*${numHT} ))

#export OMP_NUM_THREADS=${numLC}
#echo "OMP_NUM_THREADS="${OMP_NUM_THREADS}
#srun_command="srun -n${num_mpi_ranks} -c${numHT} --cpu_bind=cores"

export OMP_NUM_THREADS=${threads_per_rank}
echo "OMP_NUM_THREADS="${OMP_NUM_THREADS}
srun_command="srun -n${num_mpi_ranks} -c${numLC} --cpu_bind=cores"

echo "${srun_command} ./${EXEC} ${Input}"

${srun_command} ./${EXEC} ${Input}
