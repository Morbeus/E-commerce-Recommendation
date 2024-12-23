import pandas as pd
import json
from openai import OpenAI
from sklearn.model_selection import train_test_split
import time
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load and prepare data
data = []
with open('Dataset for DS Case Study.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Function to create training examples
def create_training_examples(df):
    # Filter for high-quality reviews (rating >= 4 and helpful votes >= 20)
    df['helpful_votes'] = df['helpful'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0)
    high_quality_df = df[
        (df['overall'] >= 4.0) & 
        (df['helpful_votes'] >= 20)
    ]
    
    training_data = []
    
    # Group by user
    user_groups = high_quality_df.groupby('reviewerID')
    
    for user_id, user_data in user_groups:
        # Sort user's reviews by time
        user_data = user_data.sort_values('unixReviewTime')
        
        if len(user_data) >= 5:  # Only use users with at least 5 reviews
            # Use earlier reviews to predict later ones
            for i in range(1, len(user_data)):
                previous_reviews = user_data.iloc[:i]
                current_review = user_data.iloc[i]
                
                # Create the prompt
                prompt = f"Based on the following user reviews:\n"
                for _, prev_review in previous_reviews.iterrows():
                    prompt += f"Product: {prev_review['asin']}\n"
                    prompt += f"Rating: {prev_review['overall']}/5\n"
                    prompt += f"Review: {prev_review['reviewText']}\n\n"
                
                prompt += "Recommend a product and explain why."
                
                # Create the completion (desired output)
                completion = f"I recommend product {current_review['asin']} because it aligns with your preferences. "
                completion += f"Based on your previous reviews, you value quality and functionality. "
                completion += f"This product has received positive feedback for similar features you've appreciated before."
                
                training_data.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful product recommendation assistant."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion}
                    ]
                })
                
                # Limit to 100 training examples
                if len(training_data) >= 10:
                    return training_data
    
    return training_data

# Create training data
# print("Creating training examples...")
# training_data = create_training_examples(df)

# print(f"\nNumber of training examples created: {len(training_data)}")

# # Save training data to a JSONL file
# with open('training_data.jsonl', 'w') as f:
#     for entry in training_data:
#         json.dump(entry, f)
#         f.write('\n')

# # Add this after creating the training data file to verify format
# with open('training_data.jsonl', 'r') as f:
#     first_line = f.readline()
#     print("\nSample training data:")
#     print(first_line)

def create_fine_tune():
    try:
        print("Creating fine-tune job...")
        
        # Upload the file
        training_file = client.files.create(
            file=open('training_data.jsonl', 'rb'),
            purpose='fine-tune'
        )
        
        print(f"File uploaded with ID: {training_file.id}")
        
        # Wait for file to be processed
        while True:
            file_status = client.files.retrieve(training_file.id)
            print(f"File status: {file_status.status}")
            if file_status.status == "processed":
                break
            time.sleep(5)
        
        # Create the fine-tuning job
        job = client.fine_tuning.jobs.create(
            training_file=training_file.id,
            model="gpt-3.5-turbo-1106",
            hyperparameters={
                "n_epochs": 2
            }
        )
        
        print(f"Fine-tuning job created: {job.id}")
        return job.id
    
    except Exception as e:
        print(f"Error creating fine-tune: {str(e)}")
        return None

def monitor_fine_tune(job_id):
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            print(f"Status: {job.status}")
            
            if job.status == 'failed':
                # Get error message
                print(f"Job failed with error: {job.error}")
                break
            elif job.status == 'succeeded':
                break
                
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Error monitoring fine-tune: {str(e)}")
            break
    
    return job.status == 'succeeded'

def get_personalized_recommendation(user_id, df, model_id):
    try:
        # Get user's review history
        user_reviews = df[df['reviewerID'] == user_id].sort_values('unixReviewTime')
        
        if len(user_reviews) == 0:
            return "No review history found for this user."
        
        # Create the prompt
        prompt = f"Based on the following user reviews:\n"
        for _, review in user_reviews.iterrows():
            prompt += f"Product: {review['asin']}\n"
            prompt += f"Rating: {review['overall']}/5\n"
            prompt += f"Review: {review['reviewText']}\n\n"
        
        prompt += "Please recommend a product and provide a detailed explanation why it would be suitable for this user."
        
        # Get recommendation from the model
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful product recommendation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating recommendation: {str(e)}"

def test_recommendation_system(model_id):
    # Get a sample user
    sample_user = df['reviewerID'].iloc[2]
    
    print(f"Generating recommendation for user: {sample_user}")
    print("\nUser's review history:")
    user_reviews = df[df['reviewerID'] == sample_user]
    for _, review in user_reviews.iterrows():
        print(f"Product: {review['asin']}")
        print(f"Rating: {review['overall']}/5")
        print(f"Review: {review['reviewText'][:200]}...")
        print("-" * 80)
    
    print("\nGenerated Recommendation:")
    recommendation = get_personalized_recommendation(sample_user, df, model_id)
    print(recommendation)

if __name__ == "__main__":
    # Create and start fine-tuning
    # job_id = create_fine_tune()
    # list_of_jobs = client.fine_tuning.jobs.list()
    # print(list_of_jobs)
    
    # if job_id:
    #     # Monitor fine-tuning progress
    #     success = monitor_fine_tune(job_id)
        
    #     if success:
    #         print("Fine-tuning completed successfully!")
    #         # Get the model ID from the fine-tuning job
    #         job = client.fine_tuning.jobs.retrieve(job_id)
    #         fine_tuned_model = job.fine_tuned_model
            
    #         # Test the system
    #         test_recommendation_system(fine_tuned_model)
    #     else:
    #         print("Fine-tuning failed.")
    # completion = client.chat.completions.create(
    # model="ft:gpt-3.5-turbo-1106:movius-software-private-limited::AhCKJLms",
    # messages=[
    #     {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Recommend a tire gauge for me"}
    #     ]
    # )
    fine_tuned_model = "ft:gpt-3.5-turbo-1106:movius-software-private-limited::AhCKJLms"
    test_recommendation_system(fine_tuned_model)
    # print(completion)