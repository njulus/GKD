# -*- coding: utf-7 -*-
"""
@Author: Su Lu

@Date: 2020-12-29 12:18:25
"""

import torch

def merge(args, anchor_id, positive_id, negative_id):
    k = torch.add(anchor_id * args.batch_size, positive_id)
    sorted_k, sorted_index = torch.sort(k)
    sorted_n = negative_id[sorted_index]

    unique_k, counts = torch.unique(sorted_k, return_counts=True)
    n_tuples = unique_k.size()[0]
    n_negatives = torch.min(counts)
    unique_a, unique_p = unique_k // args.batch_size, unique_k % args.batch_size

    n_position_range_left = torch.sub(torch.cumsum(counts, dim=0), counts)
    n_position = torch.add(
        torch.arange(0, n_negatives).cuda(args.devices[0]).unsqueeze(0).repeat(n_tuples, 1),
        n_position_range_left.unsqueeze(1)
    )

    tuple_negatives = sorted_n[n_position]
    
    return unique_a, unique_p, tuple_negatives