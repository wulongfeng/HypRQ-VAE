import argparse
import json
import os
import sys

import torch
import transformers
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from collections import defaultdict
from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results, get_detailed_topk_results
# import os

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'
from types import SimpleNamespace

def test_ddp(args):

    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path)
    if args.lora:
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,              
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    # assert model.config.vocab_size == len(tokenizer)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True)

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)


    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             sampler=ddp_sampler, num_workers=2, pin_memory=True)

    if local_rank == 0:
        print("data num:", len(test_data))

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []
    with torch.no_grad():
        for prompt_id in prompt_ids:
            if local_rank == 0:
                print("Start prompt: ",prompt_id)

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            merged_predction_results = []

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                #print(f'input:{inputs}')
                #print(f'targets:{targets}')
                bs = len(targets)
                num_beams = args.num_beams
                while True:
                    try:
                        output = model.module.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=10,
                            prefix_allowed_tokens_fn=prefix_allowed_tokens,
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            early_stopping=True,
                        )
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        print("Out of memory!")
                        num_beams = num_beams -1
                        print("Beam:", num_beams)
                    except Exception:
                        raise RuntimeError

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                #topk_res = get_topk_results(output, scores, targets, num_beams,
                #                            all_items=all_items if args.filter_items else None)
                detailed_res = get_detailed_topk_results(output, scores, targets, num_beams,
                                            all_items=all_items if args.filter_items else None)
                topk_res = detailed_res['results']

                input_list = detailed_res['input_list']
                pred_list = detailed_res['pred_list']

                # the length of them equals to the value of batch size
                assert len(input_list) == len(pred_list)
                assert len(input_list) == len(targets)

                combined_raw_data = []
                for i, elem in enumerate(input_list):
                    one_record = {}
                    one_record['input'] = input_list[i]
                    one_record['target'] = targets[i]
                    one_record['prediction'] = pred_list[i]
                    combined_raw_data.append(one_record)

                bs_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=bs, object_list=bs_gather_list)
                total += sum(bs_gather_list)
                res_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

                raw_data_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=combined_raw_data, object_list=raw_data_gather_list)
                #print(f'len of raw data gather:{len(raw_data_gather_list)}, raw data gather list:{raw_data_gather_list}')

                if local_rank == 0:
                    for raw_data in raw_data_gather_list:
                        merged_predction_results += raw_data

                    all_device_topk_res = []
                    for ga_res in res_gather_list:
                        all_device_topk_res += ga_res
                    batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
                    for m, res in batch_metrics_res.items():
                        if m not in metrics_results:
                            metrics_results[m] = res
                        else:
                            metrics_results[m] += res

                    if (step + 1) % 50 == 0:
                        temp = {}
                        for m in metrics_results:
                            temp[m] = metrics_results[m] / total
                        print(temp)

                dist.barrier()

            if local_rank == 0:
                for m in metrics_results:
                    metrics_results[m] = metrics_results[m] / total

                all_prompt_results.append(metrics_results)
                print("======================================================")
                print("Prompt {} results: ".format(prompt_id), metrics_results)
                print("======================================================")
                print("")

            dist.barrier()

    dist.barrier()

    if local_rank == 0:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res)/len(all_res)
            min_results[m] = min(all_res)
            max_results[m] = max(all_res)

        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")


        save_data={}
        save_data["test_prompt_ids"] = args.test_prompt_ids
        save_data["mean_results"] = mean_results
        save_data["min_results"] = min_results
        save_data["max_results"] = max_results
        save_data["all_prompt_results"] = all_prompt_results

        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)
        print("Save file: ", args.results_file)

        with open(args.results_detail_file, 'w') as f:
            json.dump(merged_predction_results, f)
        print('Saving raw predictions: ', args.results_detail_file)

def build_parser():
    p = argparse.ArgumentParser(description="LLMRec_test")
    # Keep using the helpers you already have
    p = parse_global_args(p)
    p = parse_dataset_args(p)
    p = parse_test_args(p)

    p.set_defaults(
        # checkpoint & model
        ckpt_path="./ckpt/ml_1m_tiger/",
        base_model="huggyllama/llama-7b",

        # data
        dataset="ml_1m",
        data_path="../data",
        index_file=".index_tiger.json",
        inter_file=".inter.json",

        # output files
        results_file="./results/ml_1m/ddp_results_tiger.json",
        results_detail_file="./results/ml_1m/ddp_details_tiger.json",

        # test hyper-parameters
        test_batch_size=4,
        num_beams=20,
        test_prompt_ids=0,
    )
    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")

    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test_ddp(args)

