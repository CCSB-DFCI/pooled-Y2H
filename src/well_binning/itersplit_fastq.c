#include <zlib.h>
#include <stdio.h>
#include "kseq.h"
KSEQ_INIT(gzFile, gzread)


int main(int argc, char *argv[])
{
	/* 
	split an input fastq.gz file into n fastq output files 
	prints total number of reads in input file
	Usage: itersplit_fastq <in.fastq.gz> <n_files>
	*/
	gzFile fp;
	kseq_t *seq;
	int l;

    if (argc == 1) {  
        fprintf(stderr, "Usage: %s <in.fastq.gz> <n_files>\n", argv[0]);  
        return 1;  
    }  


	fp = gzopen(argv[1], "r");

	//store input n_jobs argument as int
	int n_jobs;
	sscanf(argv[2], "%d", &n_jobs);

	seq = kseq_init(fp);

	//define file array
	FILE *filearray[n_jobs];

	//define output format/filenames
	char fname_format[] = "fastq_splitt_%03d.fastq";
	char fname[sizeof(fname_format) + 2];

	//open files in append mode
	for (int i=0; i < n_jobs; i++) {
		snprintf(fname, sizeof(fname), fname_format, i);
		filearray[i] = fopen(fname, "a"); 
	}

	//iterate thru input file, appending reads to split fastq files
	int count = 0;
	int total_reads = 0;
	FILE *file;
	while ((l = kseq_read(seq)) >= 0) {

		file = filearray[count];
		fprintf(file, "@");
		fprintf(file, "%s\n", seq->name.s);
		fprintf(file, "%s\n+\n", seq->seq.s);
		fprintf(file, "%s\n", seq->qual.s);
		count++;
		total_reads++;
		if (count % n_jobs == 0) { 
			count = 0;
		}
	}

	//close files
	for (int i=0; i < n_jobs; i++) { 
		file = filearray[i];
		fclose(file);
	}

	printf("%d",total_reads);
	kseq_destroy(seq);
	gzclose(fp);
	return 0;
}
