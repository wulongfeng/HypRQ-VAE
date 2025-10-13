import json
from collections import defaultdict, Counter
from math import ceil
import sys



def load_mapping(mapping_file):
   nid2sid, sid2nid = {}, {}
   with open(mapping_file, 'r') as f:
       result = json.load(f)
       for k in result:
           sid = "".join(result[k])
           nid2sid[k] = sid
           sid2nid[sid] = k
   return nid2sid, sid2nid




def split_head_tail(inter):
   item_popularity = Counter()
   for iid_list in inter.values():
       item_popularity.update(iid_list)


   sorted_items = item_popularity.most_common()     # already sorted desc
  
   # --- split 20 % / 80 % ---
   n_items      = len(sorted_items)
   top_n        = max(1, ceil(0.20 * n_items))      # always at least 1
   top_20_keys  = [k for k, _ in sorted_items[:top_n]]
   tail_80_keys = [k for k, _ in sorted_items[top_n:]]
  
   top_20_set  = set(top_20_keys)
   tail_80_set = set(tail_80_keys)
  
   return top_20_set, tail_80_set




def ratio_head_tail(preds, top_20_set, tail_80_set):
   head_num = len(set(preds).intersection(top_20_set))
   tail_num = len(set(preds).intersection(tail_80_set))
   #if head_num + tail_num != len(preds):
   #    print(f'head: {head_num}, tail: {tail_num}, and len of pred:{len(preds)}, preds:{preds}')
   return head_num/(head_num+tail_num), tail_num/(head_num+tail_num)


def ratio_head_tail_towardsTail(preds, top_20_set, tail_80_set):
   head_num = len(set(preds).intersection(top_20_set))
   tail_num = len(set(preds).intersection(tail_80_set))
   #if head_num + tail_num != len(preds):
   #    print(f'head: {head_num}, tail: {tail_num}, and len of pred:{len(preds)}, preds:{preds}')
   return (len(preds)-tail_num)/(len(preds)), tail_num/(len(preds))




def calculate_ratios(prediction_file, head_items, tail_items):
   """Calculate head/tail ratios in predictions"""
   results = {
       'target_head': {'head_ratio_5': 0, 'tail_ratio_5': 0, 'head_ratio_10': 0, 'tail_ratio_10': 0},
       'target_tail': {'head_ratio_5': 0, 'tail_ratio_5': 0, 'head_ratio_10': 0, 'tail_ratio_10': 0}
   }
   num_head, num_tail = 0, 0
   with open(prediction_file, 'r') as f:
       for line in f:
           data = json.loads(line.strip())
           target_semantic = data['target'][0]
           preds_semantic = data['prediction'][0]
          
           top_results_5 = preds_semantic[:5]
           top_results_10 = preds_semantic[:10]
           head_ratio_5, tail_ratio_5 = ratio_head_tail_towardsTail(top_results_5, head_items, tail_items)
           head_ratio_10, tail_ratio_10 = ratio_head_tail_towardsTail(top_results_10, head_items, tail_items)


           if target_semantic in head_items:
               num_head += 1
               results['target_head']['head_ratio_5'] += head_ratio_5
               results['target_head']['tail_ratio_5'] += tail_ratio_5
               results['target_head']['head_ratio_10'] += head_ratio_10
               results['target_head']['tail_ratio_10'] += tail_ratio_10
           else:
               num_tail += 1
               results['target_tail']['head_ratio_5'] += head_ratio_5
               results['target_tail']['tail_ratio_5'] += tail_ratio_5
               results['target_tail']['head_ratio_10'] += head_ratio_10
               results['target_tail']['tail_ratio_10'] += tail_ratio_10
   # Calculate ratios
   #print(results)
   for target_type in results:
       # if target_type == 'target_head':
       #     valid_num = num_head
       # else:
       #     valid_num = num_tail
       if target_type == 'target_tail':
           valid_num = num_tail
       else:
           valid_num = num_head


       results[target_type]['head_ratio_5'] /= valid_num
       results[target_type]['tail_ratio_5'] /= valid_num
       results[target_type]['head_ratio_10'] /= valid_num
       results[target_type]['tail_ratio_10'] /= valid_num
   print('Number of head items in target:{}, tail items in target:{}'.format(num_head, num_tail))


   return results






def main():
   # File paths - update these according to your file locations
   dataset = sys.argv[1]
   mapping_file = sys.argv[2]
  
   prediction_file = sys.argv[3]  # jsonl with 'target' and 'preds' fields
   # Load data
   inter_file = '../data/{}/{}.inter.json'.format(dataset, dataset)
   nid2sid, sid2nid = load_mapping(mapping_file)
   inter = json.load(open(inter_file,'rb'))
   # This version
   top20keys, tail80_keys = split_head_tail(inter)
   print(f'len of head items: {len(top20keys)}')
   print(f'len of tail items: {len(tail80_keys)}')


   top_tail_len = len(set(top20keys).intersection(tail80_keys))
   print(f'len of intersection between head and tail items: {top_tail_len}')


   head_items_sid = set([nid2sid[str(key)] for key in top20keys])
   tail_items_sid = set([nid2sid[str(key)] for key in tail80_keys])
   top_tail_sid_len = len(set(head_items_sid).intersection(tail_items_sid))
   print(f'len of intersection between head and tail sid items: {top_tail_sid_len}')
   # Calculate ratios
   results = calculate_ratios(prediction_file, head_items_sid, tail_items_sid)
  
   # Print results
   print("Head/Tail Item Ratios in Predictions:")
   print("=" * 40)
  
   for target_type, stats in results.items():
       print(f"\n{target_type.replace('_', ' ').title()}:")
       print(f"Ratio for top-5 predictions: Head:({stats['head_ratio_5']:.4f}), and Tail:({stats['tail_ratio_5']:.4f})")
       print(f"Ratio for top-10 predictions: Head: ({stats['head_ratio_10']:.4f}), and Tail:({stats['tail_ratio_10']:.4f})")




if __name__ == "__main__":
   main()


