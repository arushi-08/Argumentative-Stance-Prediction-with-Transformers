import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

def preprocess_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#', '', tweet)
    # Remove special characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Remove special characters
    tweet = re.sub(r'\s+', ' ', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    return tweet

dataset='gun_control'
df = pd.read_csv(f'./data/{dataset}_test.csv')
df_train = pd.read_csv(f'./data/{dataset}_train.csv')
df['tweet_text_cleaned'] = df['tweet_text'].apply(preprocess_tweet)
df_train['tweet_text_cleaned'] = df_train['tweet_text'].apply(preprocess_tweet)


# Load the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def get_similar_prompt_idx(text, sentences):
    # Compute the embeddings for the text and sentences
    text_embedding = model.encode(text, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    # Compute the cosine similarity between the text and each sentence
    cosine_scores = util.pytorch_cos_sim(text_embedding, sentence_embeddings)[0]
    # Find the index of the most similar sentence
    most_similar_index = cosine_scores.argmax()
    # Print the most similar sentence
    return most_similar_index


predictions = []

for idx, row in df.iterrows():
    
    context_idx = get_similar_prompt_idx(row['tweet_text_cleaned'], df_train['tweet_text_cleaned'].tolist())
    context = df_train['tweet_text_cleaned'].iloc[context_idx]
    label_for_context = df_train['stance'].iloc[context_idx]
    
    image = Image.open(
      f"./data/images/gun_control/{row['tweet_id']}.jpg"
    ).convert("RGB")
    
    prompt = f"""
    Predict support or oppose if the given text and image supports or opposes gun control topic
    example:
    {context}: {label_for_context}
    question:
    {row['tweet_text']}
    """
#     prompt = f"{context}: {label_for_context}" + row['tweet_text'] + ".\n Predict support or oppose if the given text and image supports or opposes gun control topic"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(next(model.parameters()).dtype).to(next(model.parameters()).device)

    outputs = model.generate(
      **inputs,
      do_sample=False,
      num_beams=5,
      max_length=256,
      min_length=1,
      top_p=0.9,
      repetition_penalty=1.5,
      length_penalty=1.0,
      temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    predictions.append(generated_text)
    break

print(predictions)
# df['predicted_stance_3'] = predictions
# df.to_csv('./data/gun_control_instruct_blip_predictions.csv',
#           index=False)


