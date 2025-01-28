import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub


def process_image(image):
    image = tf.image.resize(image, (224, 224))  
    image = image / 255.0  
    return image.numpy()


def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
<<<<<<< HEAD
    processed_image = np.expand_dims(processed_image, axis=0)  
=======
    processed_image = np.expand_dims(processed_image, axis=0) 
>>>>>>> bde465282bfce03323b6c96adc1f1217c0e856a8
    predictions = model.predict(processed_image)
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]
    top_k_classes = [str(index) for index in top_k_indices]
    return top_k_probs, top_k_classes


def load_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    return class_names


def main():
    parser = argparse.ArgumentParser(description='Predict the class of a flower image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')

    args = parser.parse_args()

<<<<<<< HEAD
   
=======
    
>>>>>>> bde465282bfce03323b6c96adc1f1217c0e856a8
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    
    class_names = None
    if args.category_names:
        class_names = load_class_names(args.category_names)

<<<<<<< HEAD
    
    probs, classes = predict(args.image_path, model, args.top_k)

    
    print("Probabilities:", probs)
    print("Classes:", classes)

    
=======
   
    probs, classes = predict(args.image_path, model, args.top_k)

   
    print("Probabilities:", probs)
    print("Classes:", classes)

  
>>>>>>> bde465282bfce03323b6c96adc1f1217c0e856a8
    if class_names:
        flower_names = [class_names[class_index] for class_index in classes]
        print("Flower Names:", flower_names)

if __name__ == '__main__':
    main()
