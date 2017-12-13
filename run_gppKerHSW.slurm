#!/bin/bash
#SBATCH -N 1
#SBATCH -A m1759
#SBATCH -S 4
#SBATCH -J gppKerKokkos_HSW
#SBATCH -p debug
#SBATCH -t 00:30:00 
#SBATCH -C haswell

export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#EXEC=gppKer.ex
#EXEC=gppKerFort.ex
EXEC=gppKer_gpuComplexOpenMP.ex
Input="512 2 32768 20"

num_mpi_ranks=1
numHT=1
threads_per_rank=$((32/${num_mpi_ranks}))
numLC=$(( ${threads_per_rank}*${numHT} ))

export OMP_NUM_THREADS=${threads_per_rank}
echo "OMP_NUM_THREADS="${OMP_NUM_THREADS}
srun_command="srun -n${num_mpi_ranks} -c${numLC} --cpu_bind=cores"

echo "${srun_command} ./${EXEC} ${Input}"

${srun_command} ./${EXEC} ${Input}
