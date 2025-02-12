import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *

class Easytf:
    def __init__(self, batch_size, img_height, img_width, data_dir, validation_split=0.3):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.data_dir = data_dir
        self.validation_split = validation_split
        
        # Initialize datasets
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_names = None
        self.num_classes = None
        
        # Load and prepare datasets
        self._prepare_datasets()
        
    def _prepare_datasets(self):
        # Load training dataset
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        # Load remaining dataset for validation and testing
        remaining_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        # Split remaining dataset into validation and test sets
        val_batches = tf.data.experimental.cardinality(remaining_ds) // 2
        self.val_ds = remaining_ds.take(val_batches)
        self.test_ds = remaining_ds.skip(val_batches)
        
        # Get class information
        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)
        
        # Optimize dataset performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        # Normalize the data
        normalization_layer = layers.Rescaling(1./224)
        self.train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        self.val_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y))
        self.test_ds = self.test_ds.map(lambda x, y: (normalization_layer(x), y))

    def create_model(self, custom_layers=None, base_model=None):
        inputs = Input(shape=(self.img_height, self.img_width, 3))
        if base_model is None:
            raise ValueError("base_model must be provided")
        base_model = base_model(weights='imagenet', include_top=False)
        base_model.trainable = False
        
        x = base_model(inputs, training=False)
        
        # Apply custom layers if provided
        if custom_layers:
            for layer in custom_layers:
                x = layer(x)
        else:
            # Default architecture
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs=inputs, outputs=outputs)

    def train_and_evaluate(self, model, optimizer='adamw', model_save_path='model.keras', epochs=100):
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create checkpoint callback
        checkpoint = ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        
        # Train model
        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[checkpoint]
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(self.test_ds)
        
        # Generate and save plots
        self._save_training_plots(history, epochs)
        self._save_evaluation_plots(model)
        
        return test_loss, test_accuracy, history

    def _save_training_plots(self, history, epochs):
        self._plot_metric(history, 'accuracy', 'Training and Validation Accuracy', epochs)
        self._plot_metric(history, 'loss', 'Training and Validation Loss', epochs)
        plt.savefig('validpic/training_vs_validation.png')
        plt.close()
        
        return "Training and validation plots saved successfully."

    def _plot_metric(self, history, metric, title, epochs):
        train_metric = history.history[metric]
        val_metric = history.history[f'val_{metric}']
        epochs_range = range(epochs)

        plt.figure(figsize=(20, 5))
        plt.plot(epochs_range, train_metric, label=f'Training {metric.capitalize()}')
        plt.plot(epochs_range, val_metric, label=f'Validation {metric.capitalize()}')
        plt.legend(loc='lower right' if metric == 'accuracy' else 'upper right')
        plt.title(title)

    def _save_evaluation_plots(self, model):
        # Generate predictions
        y_pred = model.predict(self.test_ds)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Get true labels
        y_true = []
        for _, labels in self.test_ds:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)

        # Confusion Matrix
        self._plot_confusion_matrix(y_true, y_pred_classes)
        plt.savefig('validpic/confusion_matrix.png')
        plt.close()

        # Precision-Recall Curve
        self._plot_precision_recall_curve(y_true, y_pred)
        plt.savefig('validpic/precision_recall_curve.png')
        plt.close()

        # F1 Curve
        self._plot_f1_curve(y_true, y_pred)
        plt.savefig('validpic/f1_curve.png')
        plt.close()
        
        return "Evaluation plots saved successfully."

    def _plot_confusion_matrix(self, y_true, y_pred_classes):
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')

    def _plot_precision_recall_curve(self, y_true, y_pred):
        precision = {}
        recall = {}
        average_precision = {}
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_pred[:, i])
            average_precision[i] = average_precision_score(y_true == i, y_pred[:, i])

        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            plt.plot(recall[i], precision[i], lw=2, 
                    label=f'{self.class_names[i]} (Average = {average_precision[i]:.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend(loc='lower left', bbox_to_anchor=(0, 1))

    def _plot_f1_curve(self, y_true, y_pred):
        f1 = []
        thresholds = np.linspace(0, 1, 101)
        for t in thresholds:
            f1_t = f1_score(y_true, (y_pred > t).argmax(axis=1), average="weighted")
            f1.append(f1_t)

        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, f1, lw=2, label="F1 Score")
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.title("F1 Curve")
        plt.legend(loc='lower left', bbox_to_anchor=(0, 1))