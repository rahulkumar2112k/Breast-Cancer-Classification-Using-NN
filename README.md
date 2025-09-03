# 🧠 Breast Cancer Classification using Neural Networks

This project uses a **Neural Network (TensorFlow + Keras)** to classify breast tumors as **Malignant (cancerous) 🔴** or **Benign (non-cancerous) 🟢** using the **Breast Cancer Wisconsin Dataset**.  
The goal is to assist in early detection of breast cancer through machine learning.

---

## 📂 Dataset
- Source: `sklearn.datasets.load_breast_cancer()`
- Features: **30 numerical features** describing tumor characteristics
- Target:
  - `0` → Malignant 🔴
  - `1` → Benign 🟢

---

## ⚙️ Project Workflow
1. **Load dataset** using `sklearn`
2. **Preprocess data**
   - Standardize features using `StandardScaler`
3. **Build Neural Network**
   - Input layer → Flatten (30 features)
   - Hidden layer → Dense (20 neurons, ReLU activation)
   - Output layer → Dense (2 neurons, Sigmoid activation)
4. **Compile Model**
   - Optimizer → Adam
   - Loss → Sparse Categorical Crossentropy
   - Metric → Accuracy
5. **Train Model**
   - Validation split: 10%
   - Epochs: 10
6. **Evaluate Model**
   - On unseen test dataset
7. **Make Predictions**
   - Convert input data → standardize → predict → classify as Malignant / Benign

---

## 📊 Model Training Results
```text
Epoch 1/10
 - accuracy: 0.6858 - val_accuracy: 0.8696
Epoch 10/10
 - accuracy: 0.9768 - val_accuracy: 0.9348
```

---
## ✅ Results on Test Data

Loss: 0.1349

Accuracy: 96.5%

---
## 🧪 Predictive System

The model can predict tumor type from new input data. Example:

```python
# Always provide input features in the same order as the training dataset.
# Changing the order can lead to wrong predictions, which is critical in medical ML.

input_data=(11.31,19.04,71.8,394.1,0.08139,0.04701,0.03709,0.0223,0.1516,0.05667,
            0.2727,0.9429,1.831,18.15,0.009282,0.009216,0.02063,0.008965,0.02183,
            0.002146,12.33,23.84,78,466.7,0.129,0.09148,0.1444,0.06961,0.24,0.06641)

# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)

# Probability distribution
print("Prediction Probabilities:", prediction)

# Final class
prediction_label = np.argmax(prediction, axis=1)[0]
print("Predicted Class Label:", prediction_label)

# Meaning
if prediction_label == 0:
    print("🔴 The tumor is **Malignant**")
else:
    print("🟢 The tumor is **Benign**")
```
---
## 🔎 Example Prediction Output  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step  

Prediction Probabilities: [[0.07830167 0.80656713]]  

Predicted Class Label: 1  

🟢 The tumor is **Benign**  

---
## ⚠️ Warning  
# Always provide input features in the same order as the training dataset.
# Changing the order can lead to wrong predictions, which is critical in medical ML.




