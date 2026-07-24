#!/usr/bin/env python3
"""Build a top-N-samples subset from an existing events.json.

Selects the N unique samples with the lowest minimum corridor distance
(each sample's severity = its best event), then writes:
  <prefix>_samples.txt   one sample token per line (severity order)
  <prefix>_scenes.txt    one scene token per line  (severity order)
  <prefix>_events.json   filtered copy of events.json with the same
                          'config' block + a refreshed 'summary' + every
                          event whose sample is in the selected set
                          (i.e. once a sample is in, all its events go in).

Usage
-----
    python tools/subset_from_events.py \\
        --events vis/interaction_occluded/events.json \\
        --top-samples 600 \\
        --out-prefix vis/interaction_occluded/top600
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--events', required=True,
                    help='Path to events.json from '
                         'viz_closest_interaction_occluded.py')
    ap.add_argument('--top-samples', type=int, required=True, metavar='N',
                    help='Target number of unique interaction-bearing samples.')
    ap.add_argument('--out-prefix', required=True, metavar='PREFIX',
                    help='Output prefix; writes <prefix>_samples.txt, '
                         '<prefix>_scenes.txt, <prefix>_events.json.')
    args = ap.parse_args()

    with open(args.events) as f:
        doc = json.load(f)
    events = doc['events']
    if not events:
        raise SystemExit('events.json has no events.')

    # Per-sample severity = minimum corridor_dist over its events.
    sample_min   = {}
    sample_scene = {}
    for e in events:
        s = e['sample_token']
        if s not in sample_min or e['corridor_dist'] < sample_min[s]:
            sample_min[s] = e['corridor_dist']
        sample_scene[s] = e['scene_token']

    ranked        = sorted(sample_min.items(), key=lambda kv: kv[1])
    top           = ranked[:args.top_samples]
    selected_set  = {s for s, _ in top}
    sample_tokens = [s for s, _ in top]
    cutoff        = top[-1][1] if top else 0.0

    # Once a sample is selected, include all its events.
    selected_events = [e for e in events if e['sample_token'] in selected_set]

    seen_c, scene_tokens = set(), []
    for s in sample_tokens:
        sc = sample_scene[s]
        if sc not in seen_c:
            seen_c.add(sc)
            scene_tokens.append(sc)

    out_dir = os.path.dirname(args.out_prefix) or '.'
    os.makedirs(out_dir, exist_ok=True)

    samples_path = f'{args.out_prefix}_samples.txt'
    scenes_path  = f'{args.out_prefix}_scenes.txt'
    events_path  = f'{args.out_prefix}_events.json'

    with open(samples_path, 'w') as f:
        f.write('\n'.join(sample_tokens) + '\n')
    with open(scenes_path, 'w') as f:
        f.write('\n'.join(scene_tokens) + '\n')

    sub_doc = {
        'config': {
            **doc.get('config', {}),
            'derived_from':       args.events,
            'top_samples_target': args.top_samples,
        },
        'summary': {
            'n_events':                len(selected_events),
            'n_unique_samples':        len(sample_tokens),
            'n_unique_scenes':         len(scene_tokens),
            'corridor_cutoff_m':       cutoff,
            'source_n_unique_samples': len(sample_min),
            'source_n_events':         len(events),
        },
        'events': selected_events,
    }
    with open(events_path, 'w') as f:
        json.dump(sub_doc, f, indent=2)

    reached = 'target reached' if len(sample_tokens) == args.top_samples \
              else (f'target {args.top_samples} not reachable — '
                    f'used all {len(sample_min)} candidate samples')
    print(f'{reached}.')
    print(f'  Selected {len(sample_tokens)} samples '
          f'({len(selected_events)} events, {len(scene_tokens)} scenes)')
    print(f'  Corridor cutoff: {cutoff:.3f} m')
    print(f'  → {samples_path}')
    print(f'  → {scenes_path}')
    print(f'  → {events_path}')


if __name__ == '__main__':
    main()
