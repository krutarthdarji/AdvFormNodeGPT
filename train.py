from dotenv import load_dotenv
import os
import time
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATA_PATH = "data/training_data.jsonl"


def main():
    # 1. Upload file
    print("Uploading file to OpenAI...")
    with open(DATA_PATH, "rb") as file:
        upload_response = client.files.create(file=file, purpose="fine-tune")
    uploaded_file_id = upload_response.id
    print("File uploaded. File ID:", uploaded_file_id)

    # 2. Create fine-tune job
    print("Creating fine-tune job...")
    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=uploaded_file_id,
        model="gpt-3.5-turbo",
    )
    fine_tune_id = fine_tune_response.id
    print("Fine-tune job created. Job ID:", fine_tune_id)

    # 3. Wait for completion
    print("Waiting for fine-tune job to complete... This can take a while.")
    status = None
    while status not in ["succeeded", "failed"]:
        time.sleep(30)  # wait before checking
        job_status = client.fine_tuning.jobs.retrieve(fine_tune_id)
        status = job_status.status
        print(f"Current status: {status}")

    print("Fine-tune job finished with status:", status)

    # 4. Save model name if successful
    if status == "succeeded":
        fine_tuned_model = job_status.fine_tuned_model
        print("Fine-tuned model name:", fine_tuned_model)
        with open("fine_tuned_model_name.txt", "w") as f:
            f.write(fine_tuned_model)


if __name__ == "__main__":
    main()
