#!/usr/bin/perl
use strict;
use warnings;

#use nvdia ug machines and run this script to check if it works for you
system("nvcc vector_add.cu -o vector_add");
#system("time ./vector_add");
#dont have permission to access gpu counters
#system("ncu -o profile ./vector_add");
