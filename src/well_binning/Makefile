all:kseq.h itersplit_fastq.c
		$(CC) -g -O2 itersplit_fastq.c -o itersplit_fastq -lz

BINDIR=/usr/local/bin

install:all 
		install itersplit_fastq $(BINDIR)

