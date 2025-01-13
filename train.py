from dotenv import load_dotenv
import os
import openai
import time

"""
Prerequisites:
1. pip install openai
2. Export OPENAI_API_KEY in your environment
   export OPENAI_API_KEY='sk-xxxx'
3. Your training data is in data/training_data.jsonl, 
   each line containing a JSON object with 'prompt' and 'completion'.
   Example line:
   {"prompt": "User wants a form with name, email", "completion": "{ \"topOptions\": [...] }"}
"""
# Make sure to set your API key (if not using environment variable):
# openai.api_key = "sk-xxxx"
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DATA_PATH = "data/training_data.jsonl"


def main():
    # 1. Validate training data format or do any needed checks
    #    e.g., check file size, chunk large data, etc. (omitted for brevity)
    # 2. Start fine-tuning
    # The recommended approach is to call openai.FineTune.create with a prepared file
    # Typically you'd first upload your file to OpenAI
    print("Uploading file to OpenAI...")
    upload_response = openai.File.create(
        file=open(DATA_PATH, "rb"), purpose="fine-tune"
    )
    uploaded_file_id = upload_response["id"]
    print("File uploaded. File ID:", uploaded_file_id)
    # 3. Create fine-tune job
    print("Creating fine-tune job...")
    fine_tune_response = openai.FineTune.create(
        training_file=uploaded_file_id,
        model="gpt-3.5-turbo",  # or 'davinci' etc. depending on availability
        # Additional hyperparams:
        # n_epochs=4,
        # batch_size=...
        # learning_rate_multiplier=...
    )
    print("Fine-tune job created. Response:", fine_tune_response)
    fine_tune_id = fine_tune_response["id"]
    # 4. Wait for completion
    print("Waiting for fine-tune job to complete... This can take a while.")
    status = None
    while status != "succeeded" and status != "failed":
        time.sleep(30)  # wait before checking
        job_status = openai.FineTune.retrieve(id=fine_tune_id)
        status = job_status["status"]
        print(f"Current status: {status}")
    print("Fine-tune job finished with status:", status)
    # 5. Retrieve the fine-tuned model name
    if status == "succeeded":
        fine_tuned_model = job_status["fine_tuned_model"]
        print("Fine-tuned model name:", fine_tuned_model)
        # Save this model name for inference
        with open("fine_tuned_model_name.txt", "w") as f:
            f.write(fine_tuned_model)


if __name__ == "__main__":
    main()
