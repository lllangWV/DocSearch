OPENAI_SETTINGS = {
    'llm_models': {
        'gpt-3.5-turbo-0125': {
            'cost': {'value_input': 0.50, 'value_output':1.50, 'unit':'per_million' },
            'contect_window': 16000
        },
        'gpt-3.5-turbo-instruct': {
            'cost': {'value_input': 1.50, 'value_output':1.50, 'unit':'per_million' },
            'max_tokens': 4000
        },
        'gpt-4o': {
            'cost': {'value_input':5.00, 'value_output':15.00, 'unit':'per_million' },
            'max_tokens': 128000
        },
        'gpt-4-turbo': {
            'cost': {'value_input':10.00, 'value_output':30.00, 'unit':'per_million' },
            'max_tokens': 128000
        },
        'gpt-4o-mini': {
            'cost': {'value_input':0.15, 'value_output':0.60, 'unit':'per_million' },
            'max_tokens': 128000
        },
        'gpt-4o-mini-2024-07-18': {
            'cost': {'value_input':0.15, 'value_output':0.60, 'unit':'per_million' },
            'max_tokens': 128000
        }
    },
    'embedding_models': {
        'text-embedding-3-small' : {
            'cost': {'value': 0.02, 'unit':'per_million' },
            'context_window': 8191},
        'text-embedding-3-large' : {
            'cost': {'value': 0.13, 'unit': 'per_million'},
            'context_window': 8191},
        'text-embedding-ada-002' : {
            'cost': {'value': 0.10, 'unit': 'per_million'},
            'context_window': 8191}
    }
}


ANTHROPIC_SETTINGS = {
    'llm_models': {
        'claude-3-5-sonnet-20240620': {
            'cost': {'value_input': 3.00, 'value_output':15.00, 'unit':'per_million' },
            'contect_window': 200000
        },
        'claude-3-opus-20240229': {
            'cost': {'value_input': 15.00, 'value_output':75.00, 'unit':'per_million' },
            'max_tokens': 200000
        },
        'claude-3-sonnet-20240229': {
            'cost': {'value_input':3.00, 'value_output':15.00, 'unit':'per_million' },
            'max_tokens': 128000
        },
        'claude-3-haiku-20240307': {
            'cost': {'value_input':0.25, 'value_output':1.25, 'unit':'per_million' },
            'max_tokens': 128000
        },

    },
    'embedding_models': {
        'voyage-large-2' : {
            'cost': {'value': 0.12, 'unit':'per_million' },
            'context_window': 16000,
            'embedding_dim': 1536
        },
        'voyage-code-2' : {
            'cost': {'value': 0.12, 'unit':'per_million' },
            'context_window': 16000,
            'embedding_dim': 1536
        },
        'voyage-2' : {
            'cost': {'value': 0.1, 'unit':'per_million' },
            'context_window': 4000,
            'embedding_dim': 1024
        },
        'voyage-lite-02-instruct' : {
            'cost': {'value': 0.12, 'unit':'per_million' },
            'context_window': 4000,
            'embedding_dim': 1024
        },
    }
}