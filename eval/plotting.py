import plotly.express as px
import pandas as pd
import ast
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
metrics_all = {'decoding': [], 'metric': [], 'value': []}

with open('texts_llama/metrics_llama_readability.json', 'r') as f:
    results = json.load(f)

    decoding_strategies = results[list(results.keys())[0]].keys()
    metrics_names = results.keys()
    
    metrics = {}
    
    for metric in metrics_names:
        for decoding_strategy in decoding_strategies:
            evals = results[metric][decoding_strategy]
            for e in evals:
                if metric == 'wsf':
                    continue
                metrics_all['decoding'].append(decoding_strategy)
                metrics_all['metric'].append(metric)
                # if metric == 'embsim' or metric == 'bleurt':
                #     metrics_all['value'].append(e * -1)
                # else:
                metrics_all['value'].append(e)
            
    # for metric in metrics.keys():
    #     df = pd.DataFrame.from_dict(metrics[metric])
    #     fig = px.line_polar(df, r="value", theta="decoding strategy", line_close=True, title=metric)
    #     fig.write_image('plots_' + str(img) + '/' + metric + '.png')

df = pd.DataFrame.from_dict(metrics_all)

fig = px.box(df, x="metric", y="value", color="decoding", notched=False, width=600, height=300)
fig.update_yaxes(range=[0, 100])


fig.write_image('plots_llama/plots_readability.png')

metrics_all = {'decoding': [], 'metric': [], 'value': []}

with open('texts_llama/metrics_llama_reference.json', 'r') as f:
    results = json.load(f)

    decoding_strategies = results[list(results.keys())[0]].keys()
    metrics_names = results.keys()
    
    metrics = {}
    
    for metric in metrics_names:
        for decoding_strategy in decoding_strategies:
            evals = results[metric][decoding_strategy]
            for e in evals:
                if metric == 'wsf':
                    continue
                metrics_all['decoding'].append(decoding_strategy)
                metrics_all['metric'].append(metric)
                # if metric == 'embsim' or metric == 'bleurt':
                #     metrics_all['value'].append(e * -1)
                # else:
                metrics_all['value'].append(e)
            
    # for metric in metrics.keys():
    #     df = pd.DataFrame.from_dict(metrics[metric])
    #     fig = px.line_polar(df, r="value", theta="decoding strategy", line_close=True, title=metric)
    #     fig.write_image('plots_' + str(img) + '/' + metric + '.png')

df = pd.DataFrame.from_dict(metrics_all)

fig = px.box(df, x="metric", y="value", color="decoding", notched=False, width=600, height=300)
fig.update_yaxes(range=[0, 0.5])

fig.write_image('plots_llama/plots_reference.png')

metrics_all = {'decoding': [], 'metric': [], 'value': []}

with open('texts_llama/metrics_llama_similarity.json', 'r') as f:
    results = json.load(f)

    decoding_strategies = results[list(results.keys())[0]].keys()
    metrics_names = results.keys()
    
    metrics = {}
    
    for metric in metrics_names:
        for decoding_strategy in decoding_strategies:
            evals = results[metric][decoding_strategy]
            for e in evals:
                if metric == 'wsf':
                    continue
                metrics_all['decoding'].append(decoding_strategy)
                metrics_all['metric'].append(metric)
                # if metric == 'embsim' or metric == 'bleurt':
                #     metrics_all['value'].append(e * -1)
                # else:
                metrics_all['value'].append(e)
            
    # for metric in metrics.keys():
    #     df = pd.DataFrame.from_dict(metrics[metric])
    #     fig = px.line_polar(df, r="value", theta="decoding strategy", line_close=True, title=metric)
    #     fig.write_image('plots_' + str(img) + '/' + metric + '.png')

df = pd.DataFrame.from_dict(metrics_all)

fig = px.box(df, x="metric", y="value", color="decoding", notched=False, width=600, height=300)
fig.update_yaxes(range=[-1, 1])

fig.write_image('plots_llama/plots_similarity.png')

metrics_all = {'decoding': [], 'metric': [], 'value': []}

with open('texts_llama/metrics_llama_diversity.json', 'r') as f:
    results = json.load(f)

    decoding_strategies = results[list(results.keys())[0]].keys()
    metrics_names = results.keys()
    
    metrics = {}
    
    for metric in metrics_names:
        for decoding_strategy in decoding_strategies:
            evals = results[metric][decoding_strategy]
            for e in evals:
                if metric == 'wsf':
                    continue
                metrics_all['decoding'].append(decoding_strategy)
                metrics_all['metric'].append(metric)
                # if metric == 'embsim' or metric == 'bleurt':
                #     metrics_all['value'].append(e * -1)
                # else:
                metrics_all['value'].append(e)
            
    # for metric in metrics.keys():
    #     df = pd.DataFrame.from_dict(metrics[metric])
    #     fig = px.line_polar(df, r="value", theta="decoding strategy", line_close=True, title=metric)
    #     fig.write_image('plots_' + str(img) + '/' + metric + '.png')

df = pd.DataFrame.from_dict(metrics_all)

fig = px.box(df, x="metric", y="value", color="decoding", notched=False, width=600, height=300)
fig.update_yaxes(range=[0, 1])

fig.write_image('plots_llama/plots_diversity.png')