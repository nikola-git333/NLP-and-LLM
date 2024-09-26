from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 


model_name = "microsoft/Phi-3-mini-4k-instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForCausalLM.from_pretrained(model_name)


user_input = str(input('Unesi tekst: '))
prompt = f'Summarize the following text : {user_input}'


pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer
) 

# parametri
params = { 
    "max_new_tokens": 100, 
    "return_full_text": False
} 

# prosljeduem parametre u obliku rijecnika funkciji pomocu **kwargs
output = pipe(prompt, **params) 
print(output[0]['generated_text']) 