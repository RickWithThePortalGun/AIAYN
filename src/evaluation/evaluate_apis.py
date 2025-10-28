import argparse
import os
import sys
import time
from tqdm import tqdm
import pandas as pd
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.loader import load_data_split
from src.utils.metrics import calculate_bleu


def translate_openai(texts: List[str], api_key: str) -> List[str]:
    """Translate texts using OpenAI API."""
    try:
        import openai
        openai.api_key = api_key
        
        translations = []
        for text in tqdm(texts, desc='OpenAI Translation'):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate the following English text to French."},
                    {"role": "user", "content": text}
                ],
                max_tokens=500,
                temperature=0.3
            )
            translations.append(response.choices[0].message.content.strip())
            time.sleep(0.1)
        
        return translations
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return []


def translate_google_cloud(texts: List[str], api_key: str) -> List[str]:
    """Translate texts using Google Cloud Translation API."""
    try:
        from google.cloud import translate_v2 as translate
        translate_client = translate.Client(api_key=api_key)
        
        translations = []
        for text in tqdm(texts, desc='Google Translation'):
            result = translate_client.translate(text, target_language='fr', source_language='en')
            translations.append(result['translatedText'])
            time.sleep(0.1)  # Rate limiting
        
        return translations
    except Exception as e:
        print(f"Error with Google Cloud API: {e}")
        return []


def translate_azure(texts: List[str], api_key: str, endpoint: str) -> List[str]:
    """Translate texts using Azure Translator."""
    try:
        from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
        
        credential = TranslatorCredential(api_key)
        client = TextTranslationClient(endpoint=endpoint, credential=credential)
        
        translations = []
        for text in tqdm(texts, desc='Azure Translation'):
            response = client.translate(content=[text], to=['fr'])
            translation = response[0] if response else None
            if translation and translation.translations:
                translations.append(translation.translations[0].text)
            else:
                translations.append("")
            time.sleep(0.1)  # Rate limiting
        
        return translations
    except Exception as e:
        print(f"Error with Azure API: {e}")
        return []


def evaluate_apis(args):
    """Evaluate translation APIs."""
    print("Evaluating translation APIs...")
    
    # Load test data
    sources, targets = load_data_split(args.data_dir, 'test')
    
    # Sample subset for evaluation
    if args.sample_size:
        sources = sources[:args.sample_size]
        targets = targets[:args.sample_size]
    
    results = {}
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    if args.api == 'openai' or args.api == 'all':
        print("\nEvaluating OpenAI...")
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            predictions = translate_openai(sources, openai_key)
            if predictions:
                bleu_score = calculate_bleu(targets, predictions)
                results['OpenAI GPT'] = bleu_score
                print(f'OpenAI BLEU Score: {bleu_score:.2f}')
        else:
            print("OpenAI API key not found. Skipping...")
    
    if args.api == 'google' or args.api == 'all':
        print("\nEvaluating Google Cloud Translation...")
        google_key = os.getenv('GOOGLE_CLOUD_API_KEY')
        if google_key:
            predictions = translate_google_cloud(sources, google_key)
            if predictions:
                bleu_score = calculate_bleu(targets, predictions)
                results['Google Cloud Translate'] = bleu_score
                print(f'Google Cloud BLEU Score: {bleu_score:.2f}')
        else:
            print("Google Cloud API key not found. Skipping...")
    
    # Evaluate Azure
    if args.api == 'azure' or args.api == 'all':
        print("\nEvaluating Azure Translator...")
        azure_key = os.getenv('AZURE_API_KEY')
        azure_endpoint = os.getenv('AZURE_ENDPOINT')
        if azure_key and azure_endpoint:
            predictions = translate_azure(sources, azure_key, azure_endpoint)
            if predictions:
                bleu_score = calculate_bleu(targets, predictions)
                results['Azure Translator'] = bleu_score
                print(f'Azure BLEU Score: {bleu_score:.2f}')
        else:
            print("Azure API credentials not found. Skipping...")
    
    # Save results
    if results:
        os.makedirs(args.results_dir, exist_ok=True)
        df = pd.DataFrame(list(results.items()), columns=['Model', 'BLEU Score'])
        results_path = os.path.join(args.results_dir, 'api_bleu_scores.csv')
        df.to_csv(results_path, index=False)
        print(f'\nResults saved to {results_path}')
    
    print("\nAPI evaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate translation APIs')
    parser.add_argument('--api', type=str, choices=['openai', 'google', 'azure', 'all'], default='all',
                       help='API to evaluate')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--sample_size', type=int, default=100, help='Number of samples to evaluate')
    
    args = parser.parse_args()
    evaluate_apis(args)
