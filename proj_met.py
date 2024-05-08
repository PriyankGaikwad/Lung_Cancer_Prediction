from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

def evaluate_model(model, test_generator):

    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    y_true = test_generator.classes
    
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    
    cm = confusion_matrix(y_true, y_pred_classes)
    
    return accuracy, precision, recall, f1, cm

if __name__ == "__main__":
    model = load_model('lung_cancer_model_vgg.h5')
    
    test_dir = 'test'

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),  
        batch_size=32,
        class_mode='categorical',
        shuffle=False  
    )
    
    batch_size = 16
    accuracy, precision, recall, f1, cm = evaluate_model(model, test_generator)

    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:")
    print(cm)
