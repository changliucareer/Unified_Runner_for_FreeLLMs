# For server with (24GB VRAM), this script is designed to 
# load models sequentially, run a test prompt, and 
# then completely clear the VRAM before the next model starts. 
# This prevents the "Out of Memory" (OOM) errors that often crash SOTA evaluation pipelines.

import os
from llm_runner import UnifiedLLMRunner
from model_registry import LLM_MODELS
import json
import csv
from output_parser import parse_appearance_output_file

def load_comments_from_json(path):
    comments = []
    with open(path, "r") as file:
        cData = json.load(file)
        for cid, comment in cData.items():
            comments.append((cid, comment))
    return comments

def load_comments_from_csv(path):
    comments = []
    with open(path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            comments.append((row["tweet_id"], row["tweet"]))
    return comments

def main():
    # input_path = "/datasets/cl0059/outputs/bluesky_replies_sampled_5000.json"
    # datasetName = "Bluesky_sampled_5000"
    input_path = "/datasets/DHDC/5000_tweets_52_women.csv"
    datasetName = "Tweets_sampled_5000"

    print("=" * 60)
    print("🚀 STARTING SOTA MODEL TEST SUITE")
    print("=" * 60)

    if input_path.endswith(".json"):
        comments = load_comments_from_json(input_path)
    elif input_path.endswith(".csv"):
        comments = load_comments_from_csv(input_path)
    
    # for debug, limit to 10 comments
    # comments = comments[:10]

    runner = UnifiedLLMRunner(task="appearance", batch_size=8, max_new_tokens=180)
    runner.run_all(comments, datasetName)

    print("\n" + "=" * 60)
    print("✅ TEST SUITE COMPLETE")
    print("=" * 60)

    # Build lookup dictionary
    comment_lookup = {cid: text for cid, text in comments}

    results_dir = "/datasets/cl0059/outputs/llm_results"
    parsed_dir = "/datasets/cl0059/outputs/parsed_csv"

    os.makedirs(parsed_dir, exist_ok=True)

    for model_name in LLM_MODELS.keys():

        input_jsonl = os.path.join(
            results_dir,
            f"{model_name}_appearance_results_{datasetName}.jsonl"
        )

        output_csv = os.path.join(
            parsed_dir,
            f"{model_name}_appearance_parsed_{datasetName}.csv"
        )

        if os.path.exists(input_jsonl):
            parse_appearance_output_file(
                input_jsonl,
                comment_lookup,
                output_csv
            )


if __name__ == "__main__":
    main()
