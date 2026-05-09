import sys

# Usage: python sample_triplets_by_rela.py <RELA_VALUE> [<NUM_SAMPLES>]
# Example: python sample_triplets_by_rela.py affects 10

def build_cui_label_dict(conso_path, language="ENG"):  # You can change language if needed
    """
    Build a dictionary mapping CUI to its preferred label from MRCONSO.RRF.
    Args:
        conso_path: Path to MRCONSO.RRF file
        language: Language code (default 'ENG')
    Returns:
        dict: {CUI: label}
    """
    cui2label = {}
    with open(conso_path, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('|')
            if len(fields) > 14:
                cui = fields[0].strip()
                lat = fields[1].strip()
                tty = fields[12].strip()  # TTY: Term Type
                is_pref = fields[6].strip()
                label = fields[14].strip()
                # Use English preferred term (TTY == 'PT')
                if lat == language and tty == 'PT':
                    cui2label[cui] = label
    return cui2label

def sample_triplets_by_rela(filepath, rela_value, num_samples=10, output_file=None, cui2label=None):
    """
    Extract sample triplets (REL, CUI1, CUI2, CUI1_LABEL, CUI2_LABEL) for a selected RELA value from MRREL.RRF.
    Args:
        filepath: Path to MRREL.RRF file
        rela_value: RELA value to filter
        num_samples: Number of samples to extract
        output_file: Optional file to save results
        cui2label: Optional dict mapping CUI to label
    """
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('|')
            if len(fields) > 10:
                rel = fields[3].strip()
                cui1 = fields[0].strip()
                cui2 = fields[4].strip()
                rela = fields[7].strip()
                if rela == rela_value:
                    cui1_label = cui2label.get(cui1, "?") if cui2label else "?"
                    cui2_label = cui2label.get(cui2, "?") if cui2label else "?"
                    samples.append((rel, cui1, cui1_label, cui2, cui2_label))
                    if len(samples) >= num_samples:
                        break
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Sample triplets for RELA='{rela_value}'\n")
            f.write(f"{'REL':<10} {'CUI1':<15} {'CUI1_LABEL':<30} {'CUI2':<15} {'CUI2_LABEL':<30}\n")
            f.write("-" * 100 + "\n")
            for rel, cui1, cui1_label, cui2, cui2_label in samples:
                f.write(f"{rel:<10} {cui1:<15} {cui1_label:<30} {cui2:<15} {cui2_label:<30}\n")
        print(f"Saved {len(samples)} samples to '{output_file}'")
    else:
        print(f"Sample triplets for RELA='{rela_value}':")
        print(f"{'REL':<10} {'CUI1':<15} {'CUI1_LABEL':<30} {'CUI2':<15} {'CUI2_LABEL':<30}")
        print("-" * 100)
        for rel, cui1, cui1_label, cui2, cui2_label in samples:
            print(f"{rel:<10} {cui1:<15} {cui1_label:<30} {cui2:<15} {cui2_label:<30}")

if __name__ == "__main__":
    # Set MRCONSO.RRF path here
    MRCONSO_PATH = "/Users/lakshankarunathilake/Documents/UMLS dataset/2025AA 2/META/MRCONSO.RRF"  # Update as needed
    # Set RELA value here
    RELA_VALUE = "chemical_or_drug_affects_cell_type_or_tissue"  # Update as needed
    # Set number of samples here
    NUM_SAMPLES = 10  # Update as needed
    FILEPATH = "/Users/lakshankarunathilake/Documents/UMLS dataset/2025AA 2/META/MRREL.RRF"
    cui2label = build_cui_label_dict(MRCONSO_PATH)
    output_file = f"sample_triplets_{RELA_VALUE}.txt"
    sample_triplets_by_rela(FILEPATH, RELA_VALUE, NUM_SAMPLES, output_file, cui2label)
