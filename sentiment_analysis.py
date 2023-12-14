from transformers import pipeline

def get_sentiments(text):
    # Load the sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Perform sentiment analysis on the input text
    results = sentiment_pipeline(text)

    # Sort the results by the highest score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Return the two highest detected sentiments
    return sorted_results[:2]

def main():
    # Get user input
    user_input = input("Enter text for sentiment analysis: ")

    # Perform sentiment analysis
    sentiments = get_sentiments(user_input)

    # Display the results
    print("\nTop detected sentiments:")
    for sentiment in sentiments:
        print(f"{sentiment['label']}: {sentiment['score']:.4f}")

if __name__ == "__main__":
    main()
