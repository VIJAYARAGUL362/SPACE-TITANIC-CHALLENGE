# Spaceship Titanic Challenge 🚀

A machine learning project to predict which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

## 📋 Project Overview

This project tackles the Kaggle Spaceship Titanic competition using multiple machine learning approaches, including traditional algorithms and deep learning with PyTorch. The goal is to predict whether a passenger was transported based on various passenger attributes and spending records.

## 🗂️ Dataset

The dataset contains information about passengers aboard the Spaceship Titanic, including:

**Categorical Features:**
- `PassengerId` - Unique identifier for each passenger
- `HomePlanet` - Planet the passenger departed from
- `CryoSleep` - Whether the passenger was in suspended animation
- `Cabin` - Cabin number (processed to extract deck information)
- `Destination` - Planet the passenger was traveling to
- `VIP` - Whether the passenger had VIP service
- `Name` - Passenger name

**Numerical Features:**
- `Age` - Age of the passenger
- `RoomService` - Amount billed for room service
- `FoodCourt` - Amount billed at the food court
- `ShoppingMall` - Amount billed for shopping mall purchases
- `Spa` - Amount billed for spa services
- `VRDeck` - Amount billed for VR deck usage

**Target Variable:**
- `Transported` - Whether the passenger was transported (True/False)

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **matplotlib** - Data visualization
- **scikit-learn** - Traditional machine learning algorithms
- **PyTorch** - Deep learning framework
- **Kaggle API** - Dataset download

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/VIJAYARAGUL362/SPACE-TITANIC-CHALLENGE.git
cd SPACE-TITANIC-CHALLENGE
```

2. Install required packages:
```bash
pip install pandas matplotlib scikit-learn torch kaggle
```

3. Set up Kaggle API credentials:
   - Download `kaggle.json` from your Kaggle account
   - Place it in the project directory when prompted

## 📊 Data Preprocessing

### Categorical Variables
- **Missing Value Imputation:**
  - `HomePlanet`: Filled with 'Earth'
  - `CryoSleep`: Filled with False
  - `Cabin`: Filled with 'G', then extracted first character (deck)
  - `Destination`: Filled with 'TRAPPIST-1e'
  - `VIP`: Filled with False

- **Encoding:**
  - Applied factorization for ordinal encoding
  - Used One-Hot Encoding with `drop='first'` parameter

### Numerical Variables
- **Missing Value Imputation:** Filled with mean values
- **Feature Scaling:** Applied StandardScaler for normalization

## 🤖 Models Implemented

### Traditional Machine Learning
1. **Support Vector Machine (SVM)** - RBF kernel
2. **Random Forest Classifier** - 300 estimators
3. **Logistic Regression**
4. **Decision Tree Classifier**

### Deep Learning
**Neural Network Architecture:**
```
Input Layer (20 features)
    ↓
Linear(20 → 32) + BatchNorm + ReLU + Dropout
    ↓
Linear(32 → 64) + BatchNorm + ReLU + Dropout
    ↓
Linear(64 → 1) + Sigmoid (Binary Classification)
```

**Training Configuration:**
- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam (lr=0.1)
- Epochs: 300
- Device: CUDA if available, else CPU

## 📈 Model Evaluation

Models are evaluated using:
- **Classification Report** (Precision, Recall, F1-Score)
- **Train-Test Split** (80-20)
- **Cross-validation** capabilities for hyperparameter tuning

## 🚀 Usage

### Running the Complete Pipeline

1. **Open the Jupyter Notebook:**
```bash
jupyter notebook SPACESHIP_TITANIC.ipynb
```

2. **Execute cells sequentially:**
   - Data loading and exploration
   - Preprocessing and feature engineering
   - Model training and evaluation
   - Test set predictions

### Key Functions

**Data Loading:**
```python
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
```

**Model Training Example:**
```python
# Traditional ML
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)

# Deep Learning
model = classification_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# Training loop...
```

## 📁 Project Structure

```
SPACE-TITANIC-CHALLENGE/
│
├── SPACESHIP_TITANIC.ipynb    # Main notebook with complete pipeline
├── train.csv                  # Training dataset (downloaded)
├── test.csv                   # Test dataset (downloaded)
├── NEUREL_NETWORK_PREDICTION.csv  # Final predictions
└── README.md                  # Project documentation
```

## 🎯 Results

The project implements multiple models with performance comparison:
- Traditional ML models provide baseline performance
- Neural network model offers competitive results
- Final predictions saved as CSV for Kaggle submission

## 🔮 Future Improvements

- [ ] Implement hyperparameter tuning with GridSearchCV
- [ ] Add ensemble methods (Voting, Stacking)
- [ ] Feature selection and engineering
- [ ] Cross-validation for more robust evaluation
- [ ] Advanced neural network architectures
- [ ] Data augmentation techniques

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Kaggle for providing the Spaceship Titanic dataset
- The open-source community for the amazing ML libraries
- Contributors and fellow data scientists for inspiration

## 📞 Contact

**Vijayaragul** - [@VIJAYARAGUL362](https://github.com/VIJAYARAGUL362)

Project Link: [https://github.com/VIJAYARAGUL362/SPACE-TITANIC-CHALLENGE](https://github.com/VIJAYARAGUL362/SPACE-TITANIC-CHALLENGE)

---
⭐ Star this repository if you found it helpful!
