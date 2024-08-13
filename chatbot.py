def chat_with_model(model_path='./model_output', model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    nlp = pipeline('text-generation', model=model, tokenizer=tokenizer)

    print("Chatbot is ready! Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = nlp(user_input, max_length=100, num_return_sequences=1)
        print(f"Chatbot: {response[0]['generated_text']}")

chat_with_model()