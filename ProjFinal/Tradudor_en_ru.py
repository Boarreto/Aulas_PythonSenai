#transformers: Biblioteca da Hugging Face usada para carregar o modelo de tradução (MarianMT).
#translit (do pacote transliterate): Transforma texto russo (alfabeto cirílico) em caracteres latinos, facilitando a leitura da pronúncia
from transformers import MarianMTModel, MarianTokenizer
from transliterate import translit

#Esse é o nome do modelo da Hugging Face que traduz de inglês (en) para russo (ru), baseado no MarianMT
# Modelo de tradução: inglês → russo
modelo_nome = 'Helsinki-NLP/opus-mt-en-ru'

# tokenizer: Transforma texto em "tokens", que o modelo consegue entender.
# model: O modelo de tradução propriamente dito
# Carregar tokenizer e modelo
tokenizer = MarianTokenizer.from_pretrained(modelo_nome)
model = MarianMTModel.from_pretrained(modelo_nome)

# Função de tradução com pronúncia
def traduzir(texto_en):
    #Transforma o texto de entrada (em inglês) em tokens
    # return_tensors="pt" Retorna os tokens como tensores PyTorch.
    tokens = tokenizer([texto_en], return_tensors="pt", padding=True)
    # O modelo processa os tokens e gera a tradução, como uma sequência de IDs de palavras
    traducao_ids = model.generate(**tokens)
    # Converte os IDs de volta para texto russo, removendo símbolos especiais (skip_special_tokens=True).
    traducao_ru = tokenizer.decode(traducao_ids[0], skip_special_tokens=True)

    # Romanizar (transliterate) o russo para alfabeto latino
    pronuncia = translit(traducao_ru, 'ru', reversed=True)

    return traducao_ru, pronuncia

# Teste no terminal
while True:
    frase = input("\nDigite uma frase em inglês (ou 'sair'): ")
    if frase.lower() == "sair":
        break
    resultado_ru, resultado_pron = traduzir(frase)
    print(f"Tradução em russo: {resultado_ru}")
    print(f"Pronúncia: {resultado_pron}")

'''
Basicamente esta biblioteca para traducao usa algo parecido com as fases de um compilador.
Se quiser enterder um pouco mais o processo de compilacao em um compilador para ter uma ideia do processo de
que este codigo e a biblioteca contida utiliza - https://www.geeksforgeeks.org/phases-of-a-compiler/.

Sao contextos totalmente diferentes, porem a ideia central e a mesma.
Em um compilador ele cria os tokens e traduz o seu codigo para linguam de maquina (0,1).
Aqui ele cria tokens de cada palavra escrita por voce, e as traduz para linguagem natural NLP.
https://huggingface.co/docs/transformers/tokenizer_summary
'''