#!/usr/bin/env bash
python umls_abbreviations_export.py "/Users/lakshankarunathilake/Documents/UMLS dataset/2025AA 2/META/MRCONSO.RRF" \
  --out output/umls_abbreviations.csv \
  --lrabr "/Users/lakshankarunathilake/Documents/UMLS dataset/2025AA 2/LRABR" \
  --mrsty "/Users/lakshankarunathilake/Documents/UMLS dataset/2025AA 2/META/MRSTY.RRF" \
  --semantic-group ANATOMY,CHEM,DEVICE,DISO,FINDING,INJURY_POISONING,LABPROC,PHYS \
  --lat ENG \
  --sab ALL

