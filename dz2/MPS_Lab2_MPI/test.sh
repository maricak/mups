#! /bin/sh
#PBS -N testjob
#PBS -o testjob.out
#PBS -e testjob.err
#PBS -l walltime=00:01:00

date
hostname
sleep 20
date
