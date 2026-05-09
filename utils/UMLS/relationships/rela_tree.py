import pandas as pd
from collections import defaultdict
import json


# MRREL.RRF file format (pipe-delimited)
# Columns: CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF|

def extract_hierarchical_relationships(filepath, max_lines=None):
    """
    Extract parent-child/hierarchical relationships from MRREL.RRF
    """
    hierarchical_rels = {
        'PAR': 'has_parent',
        'CHD': 'has_child',
        'RB': 'has_broader',
        'RN': 'has_narrower',
    }

    hierarchical_relas = [
        'isa', 'inverse_isa',
        'parent_of', 'child_of',
        'broader_than', 'narrower_than',
        'has_parent', 'has_child',
        'has_broader', 'has_narrower',
    ]

    relationships = []

    print("Reading MRREL.RRF file for hierarchical relationships...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_lines and line_num > max_lines:
                break

            if line_num % 1000000 == 0:
                print(f"Processed {line_num:,} lines...")

            fields = line.strip().split('|')

            if len(fields) > 10:
                cui1 = fields[0].strip()  # Child/Narrower concept
                rel = fields[3].strip()  # Relationship type
                cui2 = fields[4].strip()  # Parent/Broader concept
                rela = fields[7].strip()  # Specific relationship
                sab = fields[10].strip()  # Source vocabulary

                # Check if it's a hierarchical relationship
                is_hierarchical = (rel in hierarchical_rels or
                                   rela.lower() in [r.lower() for r in hierarchical_relas])

                if is_hierarchical:
                    relationships.append({
                        'child': cui1,
                        'parent': cui2,
                        'rel': rel,
                        'rela': rela,
                        'source': sab
                    })

    print(f"Found {len(relationships):,} hierarchical relationships")
    return relationships


def load_concept_names(filepath, cuis_needed):
    """
    Load concept names from MRCONSO.RRF for given CUIs
    """
    print("\nLoading concept names from MRCONSO.RRF...")
    concept_names = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000000 == 0:
                print(f"Processed {line_num:,} lines...")

            fields = line.strip().split('|')
            if len(fields) > 14:
                cui = fields[0].strip()
                if cui in cuis_needed:
                    # Get preferred name (where fields[1] == 'ENG' and fields[2] == 'P')
                    lang = fields[1].strip()
                    is_preferred = fields[6].strip()
                    name = fields[14].strip()

                    if lang == 'ENG' and is_preferred == 'P' and cui not in concept_names:
                        concept_names[cui] = name

    print(f"Loaded {len(concept_names):,} concept names")
    return concept_names


def build_tree_structure(relationships, concept_names, root_cui=None, max_depth=3):
    """
    Build a tree structure from relationships
    """
    # Build adjacency lists
    children = defaultdict(list)
    parents = defaultdict(list)

    for rel in relationships:
        children[rel['parent']].append(rel['child'])
        parents[rel['child']].append(rel['parent'])

    # Find roots (concepts with no parents) if not specified
    if root_cui is None:
        all_children = set(rel['child'] for rel in relationships)
        all_parents = set(rel['parent'] for rel in relationships)
        roots = list(all_parents - all_children)
        print(f"\nFound {len(roots)} potential root concepts")
        if roots:
            root_cui = roots[0]  # Use first root

    # Build tree recursively
    def build_subtree(cui, depth=0):
        if depth > max_depth:
            return None

        node = {
            'cui': cui,
            'name': concept_names.get(cui, cui),
            'children': []
        }

        for child_cui in children.get(cui, []):
            child_node = build_subtree(child_cui, depth + 1)
            if child_node:
                node['children'].append(child_node)

        return node

    tree = build_subtree(root_cui)
    return tree


def print_tree(node, prefix="", is_last=True):
    """
    Print tree in ASCII format
    """
    if node is None:
        return

    # Print current node
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{node['name']} ({node['cui']})")

    # Prepare prefix for children
    extension = "    " if is_last else "│   "
    new_prefix = prefix + extension

    # Print children
    children = node.get('children', [])
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_tree(child, new_prefix, is_last_child)


def save_tree_to_file(node, filepath, prefix="", is_last=True):
    """
    Save tree to text file
    """
    with open(filepath, 'a', encoding='utf-8') as f:
        if node is None:
            return

        connector = "└── " if is_last else "├── "
        f.write(f"{prefix}{connector}{node['name']} ({node['cui']})\n")

        extension = "    " if is_last else "│   "
        new_prefix = prefix + extension

        children = node.get('children', [])
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            save_tree_to_file(child, filepath, new_prefix, is_last_child)


def generate_graphviz_dot(node, dot_file):
    """
    Generate Graphviz DOT format for visualization
    """
    with open(dot_file, 'w', encoding='utf-8') as f:
        f.write("digraph UMLS_Hierarchy {\n")
        f.write("  rankdir=TB;\n")
        f.write("  node [shape=box, style=rounded];\n\n")

        def write_node(n):
            if n is None:
                return

            # Escape quotes in labels
            label = n['name'].replace('"', '\\"')
            f.write(f'  "{n["cui"]}" [label="{label}\\n({n["cui"]})"];\n')

            for child in n.get('children', []):
                f.write(f'  "{n["cui"]}" -> "{child["cui"]}";\n')
                write_node(child)

        write_node(node)
        f.write("}\n")


def analyze_hierarchy_statistics(relationships):
    """
    Analyze hierarchy statistics
    """
    children_count = defaultdict(set)
    parents_count = defaultdict(set)
    source_count = defaultdict(int)
    rel_type_count = defaultdict(int)
    rela_type_count = defaultdict(int)

    for rel in relationships:
        children_count[rel['parent']].add(rel['child'])
        parents_count[rel['child']].add(rel['parent'])
        source_count[rel['source']] += 1
        rel_type_count[rel['rel']] += 1
        if rel['rela']:
            rela_type_count[rel['rela']] += 1

    print("\n" + "=" * 80)
    print("HIERARCHY STATISTICS")
    print("=" * 80)

    print(f"\nTotal hierarchical relationships: {len(relationships):,}")
    print(f"Unique concepts involved: {len(set(children_count.keys()) | set(parents_count.keys())):,}")

    print("\n" + "-" * 80)
    print("BY SOURCE VOCABULARY:")
    print("-" * 80)
    for source, count in sorted(source_count.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {source:<20} {count:>10,}")

    print("\n" + "-" * 80)
    print("BY REL TYPE:")
    print("-" * 80)
    for rel, count in sorted(rel_type_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel:<20} {count:>10,}")

    print("\n" + "-" * 80)
    print("BY RELA TYPE (Top 20):")
    print("-" * 80)
    for rela, count in sorted(rela_type_count.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {rela:<30} {count:>10,}")

    # Find concepts with most children
    print("\n" + "-" * 80)
    print("CONCEPTS WITH MOST CHILDREN (Top 10):")
    print("-" * 80)
    sorted_by_children = sorted(children_count.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for cui, children in sorted_by_children:
        print(f"  {cui}: {len(children)} children")

    return {
        'children_count': children_count,
        'parents_count': parents_count,
        'source_count': source_count
    }


# Main execution
if __name__ == "__main__":
    umls_path = "/Users/lakshankarunathilake/Documents/UMLS dataset/2025AA 2/META"
    mrrel_file = f"{umls_path}/MRREL.RRF"
    mrconso_file = f"{umls_path}/MRCONSO.RRF"

    # Step 1: Extract hierarchical relationships
    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTING HIERARCHICAL RELATIONSHIPS")
    print("=" * 80)
    relationships = extract_hierarchical_relationships(mrrel_file)

    # Save relationships to file
    with open('hierarchical_relationships.txt', 'w', encoding='utf-8') as f:
        f.write(f"{'Child CUI':<15} {'Parent CUI':<15} {'REL':<10} {'RELA':<30} {'Source':<15}\n")
        f.write("-" * 95 + "\n")
        for rel in relationships[:10000]:  # Save first 10000
            f.write(f"{rel['child']:<15} {rel['parent']:<15} {rel['rel']:<10} {rel['rela']:<30} {rel['source']:<15}\n")
    print(f"Saved relationships to 'hierarchical_relationships.txt'")

    # Step 2: Analyze statistics
    print("\n" + "=" * 80)
    print("STEP 2: ANALYZING HIERARCHY STATISTICS")
    print("=" * 80)
    stats = analyze_hierarchy_statistics(relationships)

    # Step 3: Load concept names for visualization
    print("\n" + "=" * 80)
    print("STEP 3: LOADING CONCEPT NAMES")
    print("=" * 80)
    all_cuis = set()
    for rel in relationships:
        all_cuis.add(rel['child'])
        all_cuis.add(rel['parent'])

    concept_names = load_concept_names(mrconso_file, all_cuis)

    # Step 4: Build and visualize tree for a specific concept
    print("\n" + "=" * 80)
    print("STEP 4: BUILDING TREE VISUALIZATION")
    print("=" * 80)

    # Find a good root concept (one with many children)
    sorted_by_children = sorted(stats['children_count'].items(),
                                key=lambda x: len(x[1]), reverse=True)

    if sorted_by_children:
        root_cui = sorted_by_children[0][0]
        print(f"\nUsing root concept: {root_cui} - {concept_names.get(root_cui, 'Unknown')}")
        print(f"This concept has {len(stats['children_count'][root_cui])} direct children")

        # Build tree
        tree = build_tree_structure(relationships, concept_names, root_cui, max_depth=3)

        # Print tree to console
        print("\n" + "=" * 80)
        print("TREE VISUALIZATION (max depth = 3)")
        print("=" * 80)
        print_tree(tree)

        # Save tree to file
        with open('tree_visualization.txt', 'w', encoding='utf-8') as f:
            f.write("UMLS HIERARCHY TREE VISUALIZATION\n")
            f.write("=" * 80 + "\n\n")
        save_tree_to_file(tree, 'tree_visualization.txt')
        print(f"\nSaved tree to 'tree_visualization.txt'")

        # Generate Graphviz DOT file
        generate_graphviz_dot(tree, 'tree_visualization.dot')
        print(f"Saved Graphviz DOT file to 'tree_visualization.dot'")
        print("\nTo visualize the DOT file:")
        print("  1. Install Graphviz: brew install graphviz (Mac) or apt-get install graphviz (Linux)")
        print("  2. Run: dot -Tpng tree_visualization.dot -o tree.png")
        print("  3. Or use online viewer: https://dreampuf.github.io/GraphvizOnline/")

    # Step 5: Interactive search
    print("\n" + "=" * 80)
    print("STEP 5: SEARCH FOR SPECIFIC CONCEPT")
    print("=" * 80)
    print("\nYou can search for a specific concept by CUI or name")
    print("Example CUIs to try:")
    print("  C0011849 - Diabetes Mellitus")
    print("  C0006826 - Cancer")
    print("  C0003873 - Rheumatoid Arthritis")
    print("  C0020538 - Hypertension")