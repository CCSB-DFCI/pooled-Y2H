import os
import sys
import gzip
import regex
import joblib
import difflib
import argparse
import itertools
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from multiprocessing import Pool 
import matplotlib.pyplot as plt
import datetime

parser = argparse.ArgumentParser("Bin Oxford Nanopore fastq.gz reads by well. Input is one fastq.gz file. Output is one subsequence fastq.gz file for each well.\nExample (DB forward idx J + TERM reverse idx D) using all CPUs:\n\tpython3 bin_ont_reads.py --input-file <fastq.gz file> --cpus -1 --max-errors-primer 1 --max-errors-barcode 1 --barcode-type swim --vector db --forward-set-id J_DB --forward-primer-type DB_n_for --reverse-set-id D_AD --reverse-primer-type term_i5")
parser.add_argument("--input-file", type=str, help='Input ONT fastq.gz file', required=True)
parser.add_argument("--cpus", type=int, help='n of CPUs to use. -1 to use all available', required=True)
parser.add_argument("--max-errors-primer", type=int, help='maximum allowable errors in primer matching [adapter - [ATCG]{13} - vector sequence]. Can also mean [adapter - [ATCG]{13}] or [[ATCG]{13} -vector sequence]. Allowing more than 1 error makes pattern matching slower', required=True)
parser.add_argument("--max-errors-barcode", type=int, choices=[0, 1, 2, 3, 4, 5], help='maximum allowable errors in barcode matching [0-5]', required=True)
parser.add_argument("--barcode-type", type=str, choices=['m13', 'swim'], help='barcode type [m13, swim]. If m13, currently only M13G_for_i7 and M13G_rev_i5 are supported. If swim, --vector must be specified (ad or db)', required=True)
parser.add_argument("--vector", type=str, choices=['ad', 'db', 'adc'], help='If --barcode-type=swim, must specifiy whether ad or db')
parser.add_argument("--for-set-id", type=str, help="Set ID of forward primer (e.g. 'J_AD'). Must exactly match a value from the set_id column of the kiloseq primers table", required=True)
parser.add_argument("--for-primer-type", type=str, help="Type of forward primer (e.g. 'DB_n_for'). Must exactly match a value from the primer_type column of the kiloseq primers table", required=True)
parser.add_argument("--rev-set-id", type=str, help="Set ID of reverse primer (e.g. 'D_AD'). Must exactly match a value from the set_id column of the kiloseq primers table", required=True)
parser.add_argument("--rev-primer-type", type=str, help="Type of reverse primer (e.g. 'term_i5'). Must exactly match a value from the primer_type column of the kiloseq primers table", required=True)
args = parser.parse_args()

cwd = os.getcwd()

######create log file of args passed
script_start = datetime.datetime.now()
start_year = str(script_start.year)
start_month = str(script_start.month)
start_day = str(script_start.day)
start_hour = str(script_start.hour)
start_min = str(script_start.minute)
run_timestamp = '_'.join([start_year, start_month, start_day, start_hour, start_min])
with open(cwd + '/run_parameters_' + run_timestamp + '.log', 'w') as outfile:
	outfile.write(' '.join(sys.argv))

input_file = args.input_file
n_jobs = args.cpus

#make sure cpu number is valid
try:
	assert n_jobs <= os.cpu_count()
except:
	print ("ERROR: ", 'maximum CPUs on machine:', os.cpu_count())
	exit()

if n_jobs == -1:
	n_jobs = os.cpu_count()


forward_primer_set_id = args.for_set_id
forward_primer_type = args.for_primer_type
reverse_primer_set_id = args.rev_set_id
reverse_primer_type = args.rev_primer_type

max_primer_errors = args.max_errors_primer
max_barcode_errors = args.max_errors_barcode
BARCODE_LEN = 13
MIN_MATCHING_NTS = BARCODE_LEN - max_barcode_errors

#max barcode error dict; keys are number of mismatches. So, for example, an error of 1 means 12/13 = 0.923, so 0.9 here is the value of 1. These are used when calling difflib.get_close_matches for the cutoff parameter
#these are defined here as constants for convenience
mbe_d = {0:1.0, 1:0.9, 2:0.8, 3:0.75, 4:0.68, 5:0.6}

barcode_type = args.barcode_type

if barcode_type == 'swim':
	vector_type = args.vector
	try:
		assert (vector_type == 'ad') or (vector_type == 'db') or (vector_type == 'adc')
	except:
		print ('ERROR: When --barcode-type is swim, the vector type must be specified (ad or db)')
		exit()


###################
## split fastq 
split_fastq_prefix = 'fastq_splitt_'

print ("... splitting reads into", n_jobs, "files")

cmd = subprocess.run(['itersplit_fastq', input_file, str(n_jobs)], capture_output=True)
n_reads = int(cmd.stdout)

#check that number of created files equals n_jobs
n_split_files = len([i for i in os.listdir('./') if split_fastq_prefix in i])
try:
	assert n_split_files == n_jobs
	print("... test passed. number of split files matches n_jobs")
except:
	print("ERROR ... test failed. number of split files doesn't match n_jobs")
	exit()

###################
## bin reads by barcode

def map_ord(seq):
	return np.array(list(map(ord, seq)), dtype=np.int32)

def get_close_bar(obs_bar, exp_space_m, cutoff):
	''' given an observed barcode, search the expected barcode space for close matches '''

	#map integer code of nt
	c_ord = map_ord(obs_bar)

	#compare obs barcode to expected barcode space
	#represent result as a bool matrix
	m_m = (exp_space_m == c_ord)

	#sum m_m rowwise so each expected barcode has a score 
	s_v = np.sum(m_m, axis=1)

	#get highest score
	highest_score = s_v.max()

	#if highest score < cutoff, return empty list
	if highest_score < cutoff: return []

	#at this point the obs barcode matches at least one expected barcode with >= cutoff n of matching nts

	#get index(ices) of highest scoring barcodes
	highest_score_idx = np.where(s_v == highest_score)[0]
	
	#get number of matching expected barcodes, return dummy list if >1. This case gets classified as barcode_ambiguous by process_split_file
	n_matching_expected_bars = highest_score_idx.shape[0]
	if n_matching_expected_bars > 1: return [0,0]
	
	#at this point only one expected barcode matches the observed barcode with >= cutoff matching nts	
	highest_score_idx = highest_score_idx[0]
	
	#get matching barcode
	#matching_barcode_idx = np.
	matching_bar = [chr(int(i)) for i in exp_space_m[highest_score_idx]]
	matching_bar = ''.join(matching_bar) 
	return [matching_bar]

def process_read(seq, for_primer_reg, for_rc_primer_reg, rev_primer_reg, rev_rc_primer_reg, use_dlib=False):
		'''
		if use_dlib is True, use difflib to search barcode space (allows for imperfect barcodes) instead of get_close_bar
		'''

		#track whether the read is forward or reverse strand, as determined by both forward and reverse primers
		#if they don't agree on the orientation, then there's an orientation conflict
		for_strand_forward_primer = None	
		for_strand_reverse_primer = None	
		 
		#if two barcodes found, make sure they map to same well. If not, discard the read

		#try to find oligo seqneuces
		forw_res = regex.findall(for_primer_reg, seq) 
		rev_res = regex.findall(rev_primer_reg, seq)

		#discard read if multiple forward or reverse sequences match
		if (len(forw_res) > 1) or (len(rev_res) > 1):
			return (None, 'multiple_primer_patterns')
		#define match objects
		for_found_b, rev_found_b = None, None

		#define a single match for the forward and reverse barcode, if present
		if forw_res != []: 
			for_found_b = forw_res[0][1]
			for_strand_forward_primer = True
		else:
			forrc_res = regex.findall(for_rc_primer_reg, seq) #only search for rev comp of oligo if forwadr sequence not found
			#discard if multiple found
			if len(forrc_res) > 1:	
				return (None, 'multiple_primer_patterns')
			if forrc_res != []:
				for_found_b = forrc_res[0][1]	#take reverse complement  
				for_strand_forward_primer = False

		if rev_res != []:
			rev_found_b = rev_res[0][1]
			for_strand_reverse_primer = False
		else:
			revrc_res = regex.findall(rev_rc_primer_reg, seq) #only search for rev comp of oligo if forward sequence not found
			#discard if multiple found
			if len(revrc_res) > 1:
				return (None, 'multiple_primer_patterns')
			if revrc_res != []:
				rev_found_b = revrc_res[0][1]
				for_strand_reverse_primer = True

		if (for_strand_reverse_primer != for_strand_forward_primer) and ((for_strand_reverse_primer != None) and (for_strand_forward_primer != None)) and (vector_type != 'adc'):
			return (None, 'orientation_conflict')
			
		#now we know there's agreement about the orientation of the strand between the primer sequences that were found i.e. for_strand_reverse_primer == for_strand_forward_primer OR there's only one primer pattern in the read

		#define the expected barcode space, depending on strandedness
		if (for_strand_forward_primer == True) or (for_strand_reverse_primer == True):
			if use_dlib == False:
				expected_barcode_space = forward_strand_expected_barcodes_stacked
			else:
				expected_barcode_space = for_strand_expected_barcode_space
		else:
			if use_dlib == False:
				expected_barcode_space = reverse_strand_expected_barcodes_stacked
			else:
				expected_barcode_space = rev_strand_expected_barcode_space	

		final_well = None #the final well assigned to the read

		#now search for observed barcodes in the possible barcode space
		if (for_found_b != None) and (rev_found_b != None): #if both forward and reverse barcodes are found, try to find matches for them in the barcode space
			if use_dlib == False:
				fb_match = get_close_bar(for_found_b, expected_barcode_space, MIN_MATCHING_NTS)
				rb_match = get_close_bar(rev_found_b, expected_barcode_space, MIN_MATCHING_NTS)
			else:
				fb_match = difflib.get_close_matches(for_found_b, expected_barcode_space, cutoff=mbe_d[max_barcode_errors])
				rb_match = difflib.get_close_matches(rev_found_b, expected_barcode_space, cutoff=mbe_d[max_barcode_errors])

			####################################################
			if (len(fb_match) > 1) or (len(rb_match) > 1):
				return (None, 'bar_ambiguous')

			#if both barcodes match a barcode in the barcode space
			if (fb_match != []) and (rb_match != []):
				for_final_b = fb_match[0]
				rev_final_b = rb_match[0]
				for_well = bd[for_final_b]
				rev_well = bd[rev_final_b]
				if for_well == rev_well: #if both barcodes point to the same well
					final_well = for_well
					return (final_well, 'mapped_by_two_bar')
				else: #if they point to different wells, discard the reaed
					return (None, 'bar_conflict')

			#else if only the forward barcode matches a barcode in the barcode space
			elif fb_match != []:
				final_well = bd[fb_match[0]]
				return (final_well, 'mapped_by_one_bar')

			#else if only the reverse barcode matches a barcode in the barcode space
			elif rb_match != []:
				final_well = bd[rb_match[0]]
				return (final_well, 'mapped_by_one_bar')
			else: 
				#niether deteceted barcode matches any barcode inthe barcode space
				return (None, 'bar_ambiguous')

		#if only forward barcode found
		elif (for_found_b != None) and (rev_found_b == None):
			if use_dlib == False:
				fb_match = get_close_bar(for_found_b, expected_barcode_space, MIN_MATCHING_NTS)
			else: 
				fb_match = difflib.get_close_matches(for_found_b, expected_barcode_space, cutoff=mbe_d[max_barcode_errors]) 

			if fb_match != []:
				#discard read if multiple match
				if len(fb_match) > 1:
					return (None, 'bar_ambiguous')

				final_well = bd[fb_match[0]]
				return (final_well, 'mapped_by_one_bar')
			else: 
				#print ('obsnm', record.id, seq)
				return (None, 'obs_bar_no_match')

		#if only reverse barcode found
		elif (rev_found_b != None) and (for_found_b == None): 
			if use_dlib == False:
				rb_match = get_close_bar(rev_found_b, expected_barcode_space, MIN_MATCHING_NTS)
			else: 
				rb_match = difflib.get_close_matches(rev_found_b, expected_barcode_space, cutoff=mbe_d[max_barcode_errors])

			if rb_match != []:
				if len(rb_match) > 1:
					return (None, 'bar_ambiguous')
				final_well = bd[rb_match[0]]
				return (final_well, 'mapped_by_one_bar')
			else: 
				return (None, 'obs_bar_no_match')
		else: 
			#neither forward nor reverse barcode found
			return (None, 'bar_ambiguous')

		#this never seems to happen, but here in case
		if final_well == None: 
			#if final_well still None at this point, discard the read
			return (None, 'bar_ambiguous')


def process_split_file(read_file):
	#one function instance stores the results from one split fastq.gz file in the local p_d variable
	#i.e. p_d means "process dict" 
	#ultimately all function instance dicts get merged into one final dict of the same form

	p_d = {} #dict form :  { well : [read ids,]  }

	#initialize p_d
	for i,j in enumerate(itertools.product(range(1, 13), "ABCDEFGH"),start=1):
		p_d[j[1]+str(j[0]).zfill(2)] = []

	metadata = {'mapped_by_one_bar':0, 'mapped_by_two_bar':0, 'bar_conflict':0, 'obs_bar_no_match':0, 'multiple_primer_patterns':0, 'bar_ambiguous':0, 'orientation_conflict':0}
	
	#iterate over all reads in the split file
	with open(read_file, 'r') as infile:
		for (name,seq,quality) in SeqIO.QualityIO.FastqGeneralIterator(infile):

			final_well, category = process_read(seq, forw, forrc, rev, revrc) 

			#try to salvage reads with likely defective primer sequences using less stringent criteria; i.e. only require that the 5' end of the primer sequence is present
			#first search 5' primer fragments
			if (final_well is None) and ((category == 'bar_ambiguous') or (category == 'obs_bar_no_match')):
				final_well, category = process_read(seq, forw_fivefrag, forrc_fivefrag, rev_fivefrag, revrc_fivefrag, use_dlib=True)

				#if the 5' fragment search yields bar_ambiguous or obs_bar_no_match, try a 3' fragment search
				if (final_well is None) and ((category == 'bar_ambiguous') or (category == 'obs_bar_no_match')):
					final_well, category = process_read(seq, forw_threefrag, forrc_threefrag, rev_threefrag, revrc_threefrag, use_dlib=True)
				
			metadata[category] += 1
			if final_well is None:
				continue
			#if (well is None) and (category == 'barcode_ambiguous'):
				
			#append read id to well in output  dict
			record_id = name.split(' ')[0]
			p_d[final_well].append(record_id)

			#dump read into well-bin file
			out_text = '@' + record_id + '\n'
			out_text += seq + '\n'
			out_text += '+\n'
			out_text += quality + '\n'
			with open('./well_subseq/' + final_well + '.subseq.fastq', 'a') as outfile:
				outfile.write(out_text)

	return p_d, metadata

d = {} # {well : [reads] }
for i,j in enumerate(itertools.product(range(1, 13), "ABCDEFGH"),start=1):
    d[j[1]+str(j[0]).zfill(2)] = []


######build barcode dict of form {barcode : well}

oligosdf = pd.read_csv('./kilo_seq_primers_export_2026-03-30_140051.csv')

oligosdf = oligosdf[((oligosdf.set_id == reverse_primer_set_id) & (oligosdf.primer_type == reverse_primer_type)) | ((oligosdf.set_id == forward_primer_set_id) & (oligosdf.primer_type == forward_primer_type))].copy()
#barcode dict, mapping all barcodes to wells
bd = {}
for _,row in oligosdf.iterrows():
	bd[row.barcode] = row.well
	bd[row.barcode_rev] = row.well

forward_df = oligosdf[(oligosdf.set_id == forward_primer_set_id) | (oligosdf.primer_type == forward_primer_type)].copy()
reverse_df = oligosdf[(oligosdf.set_id == reverse_primer_set_id) | (oligosdf.primer_type == reverse_primer_type)].copy()

#barcode master dict
bar_md = {'for_bar':{}, 'for_rc_bar':{}, 'rev_bar':{}, 'rev_rc_bar':{}}

#define expected barcode space with nts as strings
for _,row in forward_df.iterrows():
	bar_md['for_bar'][row.barcode] = row.well
	bar_md['for_rc_bar'][row.barcode_rev] = row.well
	
for _,row in reverse_df.iterrows():
	bar_md['rev_bar'][row.barcode] = row.well
	bar_md['rev_rc_bar'][row.barcode_rev] = row.well
	
for_strand_expected_barcode_space = {**bar_md['for_bar'], **bar_md['rev_rc_bar']}
rev_strand_expected_barcode_space = {**bar_md['for_rc_bar'], **bar_md['rev_bar']}

#define expected barcode space with nts as integers
#convert nts to integers
forward_df['barcode_ord'] = forward_df.barcode.apply(map_ord)
forward_df['barcode_rev_ord'] = forward_df.barcode_rev.apply(map_ord)

reverse_df['barcode_ord'] = reverse_df.barcode.apply(map_ord)
reverse_df['barcode_rev_ord'] = reverse_df.barcode_rev.apply(map_ord)

for_bar_m = np.array(forward_df.barcode_ord.tolist())
for_rc_bar_m = np.array(forward_df.barcode_rev_ord.tolist())

rev_bar_m = np.array(reverse_df.barcode_ord.tolist())
rev_rc_bar_m = np.array(reverse_df.barcode_rev_ord.tolist())

forward_strand_expected_barcodes_stacked = np.vstack((for_bar_m, rev_rc_bar_m))
reverse_strand_expected_barcodes_stacked = np.vstack((for_rc_bar_m, rev_bar_m))

#template regexes for supported primer types
template_d = {  'db_for' : '(AGACGTGTGCTCTTCCGATCT([ATCG]{13})GGTCAAAGACAGTTGACTGTATCGT){e<=' + str(max_primer_errors) + '}',
				'db_for_threefrag' : '(([ATCG]{13})GGTCAAAGACAGTTGACTGTATCGT){e<=' + str(max_primer_errors) + '}',
				'db_for_fivefrag' : '(AGACGTGTGCTCTTCCGATCT([ATCG]{13})){e<=' + str(max_primer_errors) + '}',
		    'db_for_rc' : '(ACGATACAGTCAACTGTCTTTGACC([ATCG]{13})AGATCGGAAGAGCACACGTCT){e<=' + str(max_primer_errors) + '}',
			  'db_for_rc_threefrag' : '(([ATCG]{13})AGATCGGAAGAGCACACGTCT){e<=' + str(max_primer_errors) + '}',
			  'db_for_rc_fivefrag' : '(ACGATACAGTCAACTGTCTTTGACC([ATCG]{13})){e<=' + str(max_primer_errors) + '}',
				'ad_for' : '(AGACGTGTGCTCTTCCGATCT([ATCG]{13})CGATGATGAAGATACCCCACCA){e<=' + str(max_primer_errors) + '}',
				'ad_for_threefrag' : '(([ATCG]{13})CGATGATGAAGATACCCCACCA){e<=' + str(max_primer_errors) + '}',
				'ad_for_fivefrag' : '(AGACGTGTGCTCTTCCGATCT([ATCG]{13})){e<=' + str(max_primer_errors) + '}',
				'ad_for_rc' : '(TGGTGGGGTATCTTCATCATCG([ATCG]{13})AGATCGGAAGAGCACACGTCT){e<=' + str(max_primer_errors) + '}',
				'ad_for_rc_threefrag' : '(([ATCG]{13})AGATCGGAAGAGCACACGTCT){e<=' + str(max_primer_errors) + '}',
				'ad_for_rc_fivefrag' : '(TGGTGGGGTATCTTCATCATCG([ATCG]{13})){e<=' + str(max_primer_errors) + '}',

				#adc
				'adc_for' : '(GTCTCGTGGGCTCGG([ATCG]{13})AAGGTCGAATTGGGTACCGC){e<=' + str(max_primer_errors) + '}',
				'adc_for_threefrag' : '(([ATCG]{13})AAGGTCGAATTGGGTACCGC){e<=' + str(max_primer_errors) + '}',
				'adc_for_rc' : '(GCGGTACCCAATTCGACCTT([ATCG]{13})CCGAGCCCACGAGAC){e<=' + str(max_primer_errors) + '}',
				'adc_for_rc_fivefrag' : '(GCGGTACCCAATTCGACCTT([ATCG]{13})){e<=' + str(max_primer_errors) + '}',

				#term_i5 
				
				'term' : '(TCGTCGGCAGCGTC([ATCG]{13})GGAGACTTGACCAAACCTCTGGCG){e<=' + str(max_primer_errors) + '}',
				'term_threefrag' : '(([ATCG]{13})GGAGACTTGACCAAACCTCTGGCG){e<=' + str(max_primer_errors) + '}',
				'term_fivefrag' : '(TCGTCGGCAGCGTC([ATCG]{13})){e<=' + str(max_primer_errors) + '}',
				'term_rc' : '(CGCCAGAGGTTTGGTCAAGTCTCC([ATCG]{13})GACGCTGCCGACGA){e<=' + str(max_primer_errors) + '}',
				'term_rc_threefrag' : '(([ATCG]{13})GACGCTGCCGACGA){e<=' + str(max_primer_errors) + '}',
				'term_rc_fivefrag' : '(CGCCAGAGGTTTGGTCAAGTCTCC([ATCG]{13})){e<=' + str(max_primer_errors) + '}',
				'm13_for' : '(GTCTCGTGGGCTCGG([ATCG]{13})CCCAGTCACGACGTTGTAAAACG){e<=' + str(max_primer_errors) + '}',
				'm13_for_threefrag' : '(([ATCG]{13})CCCAGTCACGACGTTGTAAAACG){e<=' + str(max_primer_errors) + '}',
				'm13_for_fivefrag' : '(GTCTCGTGGGCTCGG([ATCG]{13})){e<=' + str(max_primer_errors) + '}',
				'm13_for_rc' : '(CGTTTTACAACGTCGTGACTGGG([ATGC]{13})CCGAGCCCACGAGAC){e<=' + str(max_primer_errors) + '}',
				'm13_for_rc_threefrag' : '(([ATGC]{13})CCGAGCCCACGAGAC){e<=' + str(max_primer_errors) + '}',
				'm13_for_rc_fivefrag' : '(CGTTTTACAACGTCGTGACTGGG([ATGC]{13})){e<=' + str(max_primer_errors) + '}',
				'm13_rev' : '(TCGTCGGCAGCGTC([ATGC]{13})GTAACATCAGAGATTTTGAGACAC){e<=' + str(max_primer_errors) + '}',
				'm13_rev_threefrag' : '(([ATGC]{13})GTAACATCAGAGATTTTGAGACAC){e<=' + str(max_primer_errors) + '}',
				'm13_rev_fivefrag' : '(TCGTCGGCAGCGTC([ATGC]{13})){e<=' + str(max_primer_errors) + '}',
				'm13_rev_rc' : '(GTGTCTCAAAATCTCTGATGTTAC([ATGC]{13})GACGCTGCCGACGA){e<=' + str(max_primer_errors) + '}',
				'm13_rev_rc_threefrag' : '(([ATGC]{13})GACGCTGCCGACGA){e<=' + str(max_primer_errors) + '}',
				'm13_rev_rc_fivefrag' : '(GTGTCTCAAAATCTCTGATGTTAC([ATGC]{13})){e<=' + str(max_primer_errors) + '}'
			}
				

if barcode_type == 'm13':
	forw = template_d['m13_for']
	forw_threefrag = template_d['m13_for_threefrag']
	forw_fivefrag = template_d['m13_for_fivefrag']

	forrc = template_d['m13_for_rc']
	forrc_threefrag = template_d['m13_for_rc_threefrag']
	forrc_fivefrag = template_d['m13_for_rc_fivefrag']

	rev = template_d['m13_rev']
	rev_threefrag = template_d['m13_rev_threefrag']
	rev_fivefrag = template_d['m13_rev_fivefrag']

	revrc = template_d['m13_rev_rc']
	revrc_threefrag = template_d['m13_rev_rc_threefrag']
	revrc_fivefrag = template_d['m13_rev_rc_fivefrag']

elif vector_type == 'db':
	forw = template_d['db_for']
	forw_threefrag = template_d['db_for_threefrag']
	forw_fivefrag = template_d['db_for_fivefrag']

	forrc = template_d['db_for_rc']
	forrc_threefrag = template_d['db_for_rc_threefrag']
	forrc_fivefrag = template_d['db_for_rc_fivefrag']

	rev = template_d['term']
	rev_threefrag = template_d['term_threefrag']
	rev_fivefrag = template_d['term_fivefrag']

	revrc = template_d['term_rc']
	revrc_threefrag = template_d['term_rc_threefrag']
	revrc_fivefrag = template_d['term_rc_fivefrag']

elif vector_type == 'ad':
	forw = template_d['ad_for']
	forw_threefrag = template_d['ad_for_threefrag']
	forw_fivefrag = template_d['ad_for_fivefrag']

	forrc = template_d['ad_for_rc']
	forrc_threefrag = template_d['ad_for_rc_threefrag']
	forrc_fivefrag = template_d['ad_for_rc_fivefrag']

	rev = template_d['term']
	rev_threefrag = template_d['term_threefrag']
	rev_fivefrag = template_d['term_fivefrag']

	revrc = template_d['term_rc']
	revrc_threefrag = template_d['term_rc_threefrag']
	revrc_fivefrag = template_d['term_rc_fivefrag']
	
elif vector_type == 'adc':
		#######SPECIAL CASE where we suspect 107Q-for was used with F_AD term_i5_rev
	forw = template_d['adc_for']
	forw_threefrag = template_d['adc_for_threefrag']
	forw_fivefrag = template_d['adc_for_threefrag']

	forrc = template_d['adc_for_rc']
	forrc_threefrag = template_d['adc_for_rc_fivefrag']
	forrc_fivefrag = template_d['adc_for_rc_fivefrag']

	rev = template_d['term']
	rev_threefrag = template_d['term_threefrag']
	rev_fivefrag = template_d['term_fivefrag']

	revrc = template_d['term_rc']
	revrc_threefrag = template_d['term_rc_threefrag']
	revrc_fivefrag = template_d['term_rc_fivefrag']

fastqs = [i for i in os.listdir('./') if split_fastq_prefix in i]

#check that number of reads in split files match number in original file
def check_read_counts(read_file):
	count = 0
	for record in SeqIO.parse(read_file, 'fastq'):
		count += 1
	return count

####maybe not needed
with Pool(n_jobs) as p:
	read_cnts = p.map(check_read_counts, fastqs)
	
try:
	assert sum(read_cnts) == n_reads
	print ('... test passed. sum of reads in split files matches total number of reads')
except:
	print ("... test failed. sum of reads in split files doesn't match total number of reads")
	exit()

try:
	os.mkdir('well_subseq')
except:
	pass

print ('... mapping reads to wells using', str(n_jobs), 'cpu cores', )
with Pool(n_jobs) as p:
	#process each split read file on a separate CPU core
	results = p.map(process_split_file, fastqs)

metadata_d = {'mapped_by_one_bar':0, 'mapped_by_two_bar':0, 'bar_conflict': 0, 'obs_bar_no_match':0, 'multiple_primer_patterns':0, 'bar_ambiguous':0,'orientation_conflict': 0}
print ('... creating pie chart')

for r,md_p in results:
	for feature, value in md_p.items():
		metadata_d[feature] += value
	for well, reads in r.items():
		for read in reads:
			d[well].append(read)

def autopct_format(values):
    def fmt(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{v:d}'.format(v=val)
    return fmt

cats = [k for k,v in metadata_d.items()]
vals = [metadata_d[i] for i in cats]
pied = {'categories':cats, 'values':vals}
piedf = pd.DataFrame(pied)
plt.figure(figsize=(16, 16))
plt.pie(piedf['values'], labels=piedf['categories'], autopct=autopct_format(piedf['values']))
plt.title('Barcode results:\n' + input_file + ': ' + str(n_reads) +  ' total reads')
plt.savefig('barcode_results_pie.png', dpi=300)

print ('... dumping read-well mapping object')
joblib.dump(d, 'read_well_mapping')


print ('... cleaning up files')
for split_file in fastqs:
	os.remove(split_file)

##########
##output plate map

print ("... creating read count table")
#well to well-number mapping
pos_dict = {}
for i,j in enumerate(itertools.product(range(1, 13), "ABCDEFGH"),start=1):
	if j[0]<= 9:
		strnum = '0'+str(j[0])
	else:
		strnum = str(j[0])
	pos_dict[j[1] + strnum] = i

#well-number to well mapping
rev_pos_dict = {v:k for k,v in pos_dict.items()}

l = []
for i in range(1, 9):
	n = i
	l.append(n)
	for j in range(1, 12):
		n += 8
		l.append(n)

m = []
for i in l:
	well = rev_pos_dict[i]
	n_reads = len(d[well])
	m.append(n_reads)

z = np.array(m).reshape(8, 12)

column_names = [i for i in range(1, 13)]
row_names = [i for i in 'ABCDEFGH']

df = pd.DataFrame(z, columns=column_names, index=row_names)

input_file_stripped = input_file.split('/')[-1]

writer = pd.ExcelWriter(cwd + '/' + input_file_stripped + '_well_read_counts.xlsx', engine='xlsxwriter')

df.to_excel(writer, sheet_name='mapped_read_counts')
	
writer.close()

print ('...done! total time elapsed', datetime.datetime.now() - script_start)
