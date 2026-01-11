

import csv
import statistics

def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def calc_stats(rows, phase_filter=None, cue_filter=None, cond_filter=None):
    filtered = rows
    if phase_filter: filtered = [r for r in filtered if int(r['phase']) == phase_filter]
    if cue_filter: filtered = [r for r in filtered if r['cue_type'] == cue_filter]
    if cond_filter: filtered = [r for r in filtered if r['condition'] == cond_filter]
    
    if not filtered: return None
    
    margins = [float(r['margin']) for r in filtered]
    success = []
    for r in filtered:
        if 'success_topk' in r:
             success.append(1 if r['success_topk'] == 'True' else 0)
        elif 'correct' in r:
             success.append(1 if r['correct'] == 'True' else 0)
        else:
             success.append(0)
    
    return {
        'count': len(filtered),
        'accuracy': sum(success)/len(filtered),
        'margin_mean': statistics.mean(margins),
        'margin_median': statistics.median(margins),
        'margin_min': min(margins),
        'margin_max': max(margins)
    }

def print_stats(title, stats):
    if not stats:
        print(f"[{title}] No Data")
        return
    print(f"[{title}] Acc: {stats['accuracy']:.1%} | Margin Mean: {stats['margin_mean']:.4f}, Median: {stats['margin_median']:.4f}")

def main():
    cued_data = load_csv('experimental/reports/p2_cued.csv')
    
    print("--- Detailed Stats ---")
    for cond in ['C1_JP', 'C2_EN', 'C3_MixedA']:
        # Phase 3 Prefix
        stats_p3 = calc_stats(cued_data, phase_filter=3, cue_filter='prefix', cond_filter=cond)
        print_stats(f"{cond} Prefix", stats_p3)
        
        # Phase 4 Fusion
        stats_p4 = calc_stats(cued_data, phase_filter=4, cue_filter='fusion', cond_filter=cond)
        print_stats(f"{cond} Fusion", stats_p4)

    # Cross Task A
    try:
        cross_data = load_csv('experimental/reports/p2_cross.csv')
        stats_cross = calc_stats(cross_data, phase_filter=6, cond_filter='C4_MixedB')
        print_stats(f"C4_MixedB Task A", stats_cross)
    except FileNotFoundError:
        pass
        
    # Cross Task B
    try:
        extract_data = load_csv('experimental/reports/p2_cross_extract.csv')
        stats_extract = calc_stats(extract_data, phase_filter=6, cond_filter='C4_MixedB')
        print_stats(f"C4_MixedB Task B", stats_extract)
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    main()
