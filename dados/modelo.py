import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# Baixar recursos necessários do NLTK (execute apenas uma vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# Função para pré-processamento de texto - DEVE ser idêntica à usada no treinamento
def preprocess_text(text):
    if isinstance(text, str):
        # Converter para minúsculas
        text = text.lower()
        # Remover caracteres especiais e números
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenização
        tokens = nltk.word_tokenize(text)
        # Remover stopwords em português
        stop_words = set(stopwords.words('portuguese'))
        tokens = [word for word in tokens if word not in stop_words]
        # Stemming (redução das palavras ao seu radical)
        stemmer = RSLPStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        # Reconstruir texto
        return ' '.join(tokens)
    else:
        return ''

def main():
    # Configuração de caminhos
    input_file = 'df_reviews_final.csv'
    model_file = 'modelo_categorizacao_app_bradesco.joblib'
    output_file = 'df_reviews_processasdo.csv'
    
    print(f"Iniciando processamento do arquivo: {input_file}")
    
    # Carregar o modelo
    try:
        print(f"Carregando modelo de: {model_file}")
        model = load(model_file)
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        print("Certifique-se de que o modelo foi treinado e salvo corretamente.")
        return
    
    # Carregar o dataset
    try:
        print(f"Carregando dataset de: {input_file}")
        # Tentativa com diferentes encodings e separadores
        for encoding in ['utf-8', 'latin1']:
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(input_file, encoding=encoding, sep=sep)
                    print(f"Dataset carregado com sucesso! Encoding: {encoding}, Separador: '{sep}'")
                    print(f"Dimensões: {df.shape}")
                    break
                except Exception:
                    continue
            else:
                continue
            break
        else:
            raise Exception("Nenhuma combinação de encoding/separador funcionou")
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return
    
    # Verificar as colunas disponíveis
    print("Colunas originais:")
    print(df.columns.tolist())
    
    # Guardar uma lista das colunas originais
    original_columns = df.columns.tolist()
    
    # Identificar a coluna de comentários
    review_column = 'review' if 'review' in df.columns else None
    if review_column is None:
        for col in df.columns:
            if 'review' in col.lower() or 'comentario' in col.lower():
                review_column = col
                break
    
    if review_column is None:
        print("Não foi possível identificar a coluna de comentários.")
        return
    else:
        print(f"Coluna de comentários identificada: {review_column}")
    
    # Pré-processar os comentários (em uma série temporária, não adicionar ao DataFrame)
    print("Pré-processando comentários...")
    processed_comments = df[review_column].apply(preprocess_text)
    print("Pré-processamento concluído!")
    
    # Aplicar o modelo para prever categorias
    print("Aplicando modelo para categorizar comentários...")
    try:
        # Usar os comentários processados para previsão, mas não adicioná-los ao DataFrame
        df['categoria'] = model.predict(processed_comments)
        print("Categorização concluída!")
    except Exception as e:
        print(f"Erro ao categorizar comentários: {e}")
        return
    
    # Análises básicas (apenas para exibição, não afeta o output)
    print("\nDistribuição das categorias:")
    category_counts = df['categoria'].value_counts()
    print(category_counts)
    
    # Preparar dataset para o output final
    print("\nPreparando dataset para o output final...")
    
    # Verificar colunas atuais
    print("Colunas após categorização:")
    print(df.columns.tolist())
    
    # Salvar o dataset enriquecido (original + categoria)
    print(f"Salvando dataset enriquecido em: {output_file}")
    try:
        # Garantir que apenas as colunas originais + categoria sejam salvas
        output_columns = original_columns.copy()
        if 'categoria' not in output_columns:
            output_columns.append('categoria')
        
        print(f"Colunas no output final: {output_columns}")
        df_output = df[output_columns]
        
        # Salvar no formato CSV
        df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("Dataset salvo com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar o dataset: {e}")
        print(f"Detalhes: {str(e)}")
    
    print("\nProcessamento concluído com sucesso!")
    print(f"O dataset enriquecido foi salvo em: {output_file}")

if __name__ == "__main__":
    main()