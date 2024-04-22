import json

import pandas as pd
from openai import OpenAI  # pylint: disable=import-error
from sentence_transformers import SentenceTransformer, util

client = OpenAI()

def api_cache(func, verbose=1):
    """
    A decorator function that caches the results of API calls based on the prompt and model used.
    """
    if verbose not in [0, 1, 2]:
        raise ValueError("Verbose parameter must be 0, 1, or 2.")
    cache = {}
    def wrapper(prompt, model="gpt-3.5-turbo"):
        # Use the prompt and model as the cache key
        key = (prompt, model)
        if key not in cache:
            cache[key] = func(prompt, model)
        else:
            if verbose > 0:
                print("--------------------------------------------------\
                \nFetching results ...\
                \n--------------------------------------------------")
            if verbose == 2:
                print(f"\nModel: {model}\
                \nPrompt: {prompt_preview}\
                \n--------------------------------------------------")
            # Split the prompt into words and join the first 5 or fewer
                prompt_preview = ' '.join(prompt.split()[:5]) + ('...' if len(prompt.split()) > 5 else '')
        return cache[key]
    return wrapper

# From Prompt Engineering for Developers Course
@api_cache
def get_completion(prompt, model="gpt-3.5-turbo"):
    """
    Generates a completion for the given prompt using the OpenAI Chat API.

    Parameters:
    prompt (str): The user's input prompt.
    model (str): The model to use for generating the completion. Defaults to "gpt-3.5-turbo".

    Returns:
    str: The generated completion.

    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

@api_cache
def formulate_contradiction(prompt, model="gpt-3.5-turbo-0125"):
    """
    Formulates a technical contradiction based on the given prompt using OpenAI's chat completions API.

    Args:
        prompt (str): The prompt for formulating the contradiction.
        model (str, optional): The model to use for generating the response. Defaults to "gpt-3.5-turbo-0125".

    Returns:
        str: The formulated contradiction.

    Raises:
        OpenAIError: If there is an error in the API request.

    Example:
        prompt = "How can we improve the fuel efficiency of a car?"
        contradiction = formulate_contradiction(prompt)
        print(contradiction)
        # Output: "If we increase the engine power, then the fuel efficiency decreases."
    """
    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": 'You are a helpful assistant designed to output JSON.\
            When you are prompted to define a technical contradiction, keep the form of it as below:\
            "If {feature}, then {positive_parameter} but {negative_parameter}". Detailed description of the \
            components of the contradiction are as follows: \
                {feature} - short description of the applied solution, that was meant to solve the problem.\
                {positive_parameter} - what is positively affected by the {feature}\
                {negative_parameter} - what is negatively affected by the {feature}}.\
                Use {feature}, {positive_parameter} and {negative_parameter} as JSON keys.'},
            {"role": "user", "content": prompt}
            ]
        )
    output = response.choices[0].message.content
    return json.loads(output)

def get_params(param, params_list):
    """
    Returns a list of parameters sorted by cosine similarity to the given parameter.
    
    Parameters:
        param (str): The parameter to calculate cosine similarity with.
        params_list (list): A list of parameters to compare with the given parameter.
        
    Returns:
        list: A list of parameters sorted by cosine similarity in descending order.
    """
    df = pd.DataFrame(calculate_cosine_scores(param, params_list))
    sorted_df = df.sort_values(by='cosine_similarity', ascending=False).head()
    output_list = sorted_df['text'].tolist()
    return output_list

def get_principles(improving_list, preserving_list, matrix, n_params=5):
    """
    Get a list of selected principles based on the given improving and preserving lists.

    Parameters:
    improving_list (list): A list of principles to improve.
    preserving_list (list): A list of principles to preserve.
    matrix (pandas.DataFrame): The matrix containing the principles.
    n_params (int): The number of principles to select. Default is 5.

    Returns:
    list: A list of selected principles.

    """
    principles = matrix[matrix.index.isin(improving_list)][preserving_list].values.flatten()
    principles = [x for x in principles if str(x) != 'nan']
    principles_list = [int(num) for sublist in principles for num in sublist.split(', ')]
    principles_count = {i: principles_list.count(i) for i in principles_list}
    principles_series = pd.Series(principles_count).sort_values(ascending=False)

    # Select only the first n principles
    selected_principles = principles_series.iloc[:n_params].index.tolist()
    return selected_principles

def process_contradiction(output, matrix, principles_list, parameters_list, no_params=3):
    """
    Process the contradiction based on the given output, matrix, principles list, and number of parameters.

    Args:
        output (dict): The output containing the positive and negative parameters.
        matrix (list): The matrix containing the relationship between parameters and principles.
        principles_list (list): The list of principles.
        no_params (int, optional): The number of parameters to consider. Defaults to 3.

    Returns:
        dict: A dictionary containing the selected principles and their descriptions.
    """
    parameter_to_improve = output['positive_parameter']
    parameter_to_preserve = output['negative_parameter']
    improving_list = get_params(parameter_to_improve, parameters_list)[:no_params]
    preserving_list = get_params(parameter_to_preserve, parameters_list)[:no_params]
    selected_principles = get_principles(improving_list, preserving_list, matrix)

    # Prepare dictionary in the form {number: (name, description)}
    principles_dict = {}
    for i, line in enumerate(principles_list):
        split = line.split(': ')
        key = i + 1
        value = split
        principles_dict[key] = value

    return {principles_dict[key][0]: principles_dict[key][1] for key in selected_principles}

def solve_contradiction(problem_desc, principles_dict, model="gpt-3.5-turbo-0125"):
    """
    Formulates a technical contradiction based on the given prompt using OpenAI's chat completions API.

    Args:
        prompt (str): The prompt for formulating the contradiction.
        model (str, optional): The model to use for generating the response. Defaults to "gpt-3.5-turbo-0125".

    Returns:
        str: The formulated contradiction.

    Raises:
        OpenAIError: If there is an error in the API request.

    Example:
        prompt = "How can we improve the fuel efficiency of a car?"
        contradiction = formulate_contradiction(prompt)
        print(contradiction)
        # Output: "If we increase the engine power, then the fuel efficiency decreases."
    """
    prompt = 'Problem description:/n' + problem_desc
    for key, value in principles_dict.items():
        prompt += f'/n{key}: {value}'

    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": 'You are an inventor and a TRIZ expert, \
                skilled in inventive problem solving.You will be given a problem \
                description in the form of a textual description, and a set of inventive principles.\
                Your task is to propose solutions based on the inventive principles. Format your answer in .json, \
                with principle name as a key, and the proposed solution as the value.'},
            {"role": "user", "content": prompt}
            ]
        )
    output = response.choices[0].message.content
    return json.loads(output)

def calculate_cosine_scores(text1, text2):
    """
    Calculates the cosine similarity scores between two texts using the SentenceTransformer model.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        torch.Tensor: The cosine similarity scores between the two texts.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embedding for both lists
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2).tolist()[0]

    output = {'id':[i + 1 for i in range(len(text2))],
            'text': text2,
            'cosine_similarity': cosine_scores}

    return output
