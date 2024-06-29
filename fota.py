import os
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu

# Inicjalizacja modelu InceptionV3 do ekstrakcji cech obrazu
base_model = InceptionV3(weights='imagenet')
image_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Funkcja do przetwarzania obrazu i ekstrakcji cech
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Przykładowa funkcja do generowania opisu
def generate_description(image_path):
    # Wczytanie obrazu i przetworzenie cech
    img = preprocess_image(image_path)
    features = image_model.predict(img)
    
    # Prosty model generujący opis (tu przykład, można zaimplementować bardziej zaawansowany model)
    description_dict = {
        'dog': 'A dog playing in the grass',
        'cat': 'A cat sitting on a window sill',
        'car': 'A car parked on the street'
    }
    
    # Wybór opisu na podstawie najbliższych cech (tu można użyć bardziej zaawansowanego podejścia)
    labels = ['dog', 'cat', 'car']
    similarities = [sentence_bleu([label], features.flatten()) for label in labels]
    idx = np.argmax(similarities)
    selected_label = labels[idx]
    description = description_dict[selected_label]
    
    return description

# Przykładowe użycie
if __name__ == '__main__':
    image_path = 'fota.jpg'
    if os.path.exists(image_path):
        description = generate_description(image_path)
        print(f"Generated description for '{image_path}':")
        print(description)
    else:
        print(f"Image '{image_path}' not found.")
