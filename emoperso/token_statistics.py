"""Token lengths, post coverage and text retention statistics.

TokenStatistics computes exact percentiles from per-user scalar lengths and
retention ratios. Memory grows linearly with the number of users.
"""

from collections import Counter
import numpy as np
from .data import DataContractError


class TokenStatistics:
    """Accumulate per-user lengths, retention ratios and post coverage."""

    def __init__(self):
        self.groups = {}

    def update(self, row):
        if (row.retained_characters is None or row.original_content_token_count is None
                or len(row.post_content_token_counts)!=len(row.source.posts)):
            raise DataContractError('Token statistics require certified character retention, content counts and post coverage')
        source = row.source
        content = sum(row.content_mask)
        contributing = sum(n>0 for n in row.post_content_token_counts)
        full_posts = sum(n>0 and p.user_text_end<=row.retained_characters
                         for n,p in zip(row.post_content_token_counts,source.posts))
        metrics = dict(users=1, label_eligible=int(source.label_eligible),
            token_eligible=int(row.token_eligible), supervised_token_eligible=int(row.sample_eligible),
            truncated_users=int(row.truncated), over_2048_users=int(row.original_token_count>2048),
            original_tokens=row.original_token_count, retained_tokens=len(row.input_ids),
            original_content_tokens=row.original_content_token_count, retained_content_tokens=content,
            original_characters=len(source.text), retained_characters=row.retained_characters,
            selected_posts=len(source.posts), contributing_posts=contributing, fully_retained_posts=full_posts,
            partial_posts=contributing-full_posts, zero_content_posts=len(source.posts)-contributing,
            users_with_zero_content_posts=int(contributing<len(source.posts)),
            users_with_at_most_1_contributing_post=int(contributing<=1),
            users_with_at_most_5_contributing_posts=int(contributing<=5),
            over_configured_budget_users=int(row.original_token_count>row.max_length),
            retained_registered_special_tokens=sum(row.special_tokens_mask),
            unicode_boundary_users=int(row.unicode_boundary_tokens_removed>0),
            unicode_boundary_tokens_removed=row.unicode_boundary_tokens_removed,
            label_boundary_users=int(row.label_boundary_tokens_removed>0),
            label_boundary_tokens_removed=row.label_boundary_tokens_removed)
        names = (source.dataset_id+"/all", source.dataset_id+"/"+(source.partition or "unassigned"))
        if source.label_eligible:
            names += (source.dataset_id+"/supervised",)
        for name in names:
            g = self.groups.setdefault(name, {"totals":Counter(),"original_lengths":[],"retained_lengths":[],
                "character_retention_ratios":[],"by_selected_post_count":{},"token_exclusion_reasons":Counter()})
            g["totals"].update(metrics)
            g["original_lengths"].append(row.original_token_count)
            g["retained_lengths"].append(len(row.input_ids))
            g["character_retention_ratios"].append(row.retained_characters/max(1,len(source.text)))
            g["by_selected_post_count"].setdefault(str(len(source.posts)),Counter()).update(metrics)
            g["token_exclusion_reasons"].update(row.token_exclusion_reasons)

    def result(self):
        result = {}
        for name,g in self.groups.items():
            totals = dict(g["totals"])
            describe = lambda values: {"min":min(values),"mean":float(np.mean(values)),
                "p25":float(np.percentile(values,25)),"median":float(np.median(values)),
                "p75":float(np.percentile(values,75)),"p90":float(np.percentile(values,90)),
                "p95":float(np.percentile(values,95)),"p99":float(np.percentile(values,99)),"max":max(values)}
            result[name] = {"totals":totals,"original_token_lengths":describe(g["original_lengths"]),
                "retained_token_lengths":describe(g["retained_lengths"]),
                "per_user_character_retention":describe(g["character_retention_ratios"]),
                "over_2048_fraction":totals["over_2048_users"]/totals["users"],
                "micro_character_retention":totals["retained_characters"]/max(1,totals["original_characters"]),
                "micro_content_token_retention":totals["retained_content_tokens"]/max(1,totals["original_content_tokens"]),
                "by_selected_post_count":g["by_selected_post_count"],
                "token_exclusion_reasons":g["token_exclusion_reasons"]}
        return result



def summarize_token_inputs(inputs):
    """Summarize TokenInput rows with certified retention and post coverage."""
    statistics = TokenStatistics()
    for row in inputs:
        statistics.update(row)
    return statistics.result()
