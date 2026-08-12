import transformers
from transformers import pipeline

print('transformers version:', transformers.__version__)
print('supported tasks:')
print(transformers.pipelines.get_supported_tasks())
print('text2text supported?')
try:
    p = pipeline('text2text-generation', model='google/flan-t5-base', return_full_text=False)
    print('text2text pipeline created:', type(p))
    print('sample output:', p('Explain RAG in one sentence.'))
except Exception as e:
    print('text2text pipeline error:', repr(e))

print('text-generation supported?')
try:
    p = pipeline('text-generation', model='google/flan-t5-base', return_full_text=False)
    print('text-generation pipeline created:', type(p))
    print('sample output:', p('Explain RAG in one sentence.'))
except Exception as e:
    print('text-generation pipeline error:', repr(e))
