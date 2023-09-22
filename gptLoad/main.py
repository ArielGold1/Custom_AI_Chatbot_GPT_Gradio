# Import necessary libraries and modules
if __name__ == '__main__':
    from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
    from langchain import OpenAI
    import gradio as gr
    import os

    # Set the OpenAI API key (Make sure not to expose API keys in public repositories)
    os.environ["OPENAI_API_KEY"] = 'your open api key should be here'


    # Function to construct and save the index
    def construct_index(directory_path):
        max_input_size = 4096
        num_outputs = 512
        max_chunk_overlap = 20
        chunk_size_limit = 600

        # Create a prompt helper for text generation
        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        # Initialize an LLMPredictor with OpenAI's GPT model
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="chatgpt", max_tokens=num_outputs))

        # Load and process documents from a directory
        documents = SimpleDirectoryReader(directory_path).load_data()

        # Create a GPTSimpleVectorIndex
        index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        # Save the index to disk
        index.save_to_disk('index.json')

        return index


    # Function for chatbot interaction
    def chatbot(input_text):
        # Load the index from disk
        index = GPTSimpleVectorIndex.load_from_disk('index.json')

        # Query the index and get a response
        response = index.query(input_text, response_mode="compact")
        return response.response


    # Create a Gradio interface for the chatbot
    iface = gr.Interface(fn=chatbot,
                         inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                         outputs="text",
                         title="Custom-trained AI Chatbot")

    # Construct the index
    index = construct_index(r'itemsExcel.csv')

    # Launch the Gradio interface
    iface.launch(share=True)
