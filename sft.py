import json
import argparse
import os 
import datetime
import time
from tqdm import tqdm
import jsonlines
import numpy as np
from openai import OpenAI
from openai import AzureOpenAI

def call_gpt(data, args):
    print(os.getenv("AZURE_OPENAI_ENDPOINT"))
    client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), api_key=os.getenv("AZURE_OPENAI_API_KEY"), api_version="2024-02-01")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": data,
            }
        ],
        model="GPT-4o",
    )
    return chat_completion.choices[0].message.content

def parse_completion(completion):
    # Split the completion into the math problem and solution
    problem_start = "Math problem:"
    solution_start = "Solution:"

    problem = completion.split(problem_start)[1].split(solution_start)[0].strip()
    solution = completion.split(solution_start)[1].strip()

    return problem, solution

def generation(args, data_path="../test_data/test_all.jsonl"):
    input_data = []

    with open(args.input_file1, 'r') as reader:
        input_data = json.load(reader)

    try:
        saved_idx = list(np.load(args.saved_idx).astype(int))
    except FileNotFoundError:
        saved_idx = []

    annotated = []
    for i in range(len(input_data)):
        if i in saved_idx:
            continue

        input_instance = input_data[i]
        
        with open("generation_instruction.txt", "r") as f:
            text = f.read()

        adjacency_str = json.dumps(input_instance["adjacency"])
        dependency_str = json.dumps(input_instance["dependency"])

        print(dependency_str)
        print(adjacency_str)

        text = text.replace("||adjacency||", adjacency_str)
        text = text.replace("||dependency||", dependency_str)

        try:
            completion = call_gpt(text, args)
            problem, solution = parse_completion(completion)
        except Exception as error:
            print(f"Error occurred: {error}")
            continue

        annotated_instance = {
            "math_problem": problem,
            "solution": solution
        }
        annotated.append(annotated_instance)
                
        with open(f"{args.output_file}.json", mode='a') as writer:
            json.dump(annotated_instance, writer)
            writer.write("\n")

        # Save index after processing each instance
        saved_idx.append(i)
        np.save(args.saved_idx, np.array(saved_idx))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file1", default="adjacency_dependency_100_instances.json")
    parser.add_argument("--output_file", default="output_100")
    parser.add_argument("--saved_idx", default="saved_idx.npy")
    args = parser.parse_args()

    generation(args)
