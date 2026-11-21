# last-task
🚢 Interactive Titanic Survival Prediction web app built with Streamlit and a Decision Tree Classifier, featuring real-time predictions from passenger inputs.
# 🚢 Titanic Survival Prediction Web App

## 📌 Project Overview

This project is an interactive **Titanic Survival Prediction Web Application** built using **Python, Scikit-Learn, Joblib, and Streamlit**.

The application uses a trained **Decision Tree Classifier** to predict whether a Titanic passenger is likely to survive based on passenger information such as passenger class, gender, age, family details, and fare.

The trained machine learning model is loaded using **Joblib**, and users can enter passenger details through an interactive Streamlit interface to receive a live prediction.

---

## 🎯 Objective

The main objective of this project is to deploy a trained machine learning model as an interactive web application.

This project demonstrates:

- Loading a trained machine learning model
- Using Joblib for model persistence
- Building an interactive Streamlit interface
- Accepting user inputs through sliders and selection controls
- Generating real-time predictions
- Preparing a machine learning application for deployment

---

## ✨ Features

- 🚢 Interactive Titanic survival prediction
- 🎛️ User-friendly input sliders and selection boxes
- 🤖 Decision Tree Classifier
- ⚡ Real-time prediction
- 💾 Model loading using Joblib
- 🚀 Streamlit-based web interface
- 📊 Displays the entered passenger information
- 🔒 Does not collect personal information
- ☁️ Ready for deployment using Streamlit Community Cloud

---

## 🧠 Machine Learning Model

The application uses a **Decision Tree Classifier** trained on the Titanic dataset.

### Model Configuration

- Algorithm: Decision Tree Classifier
- Train/Test Split: 80/20
- Random State: 42
- Maximum Tree Depth: 5
- Target Variable: `Survived`

### Input Features

The model uses the following features in the same order as during training:

| Feature | Description |
|---|---|
| `Pclass` | Passenger class |
| `Sex` | Passenger gender |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Passenger ticket fare |

### Data Preprocessing

- Missing `Age` values are replaced with the median age.
- `Sex` is converted into numerical values:
  - Male → `0`
  - Female → `1`
- Feature order is kept consistent between model training and prediction.

---

## 🖥️ Application Interface

The application allows users to enter:

- Passenger Class
- Sex
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare

After entering the information, the user can click:

**🔮 Predict Survival**

The application then displays either:

> 🎉 The passenger is likely to SURVIVE.

or

> ⚠️ The passenger is likely to NOT SURVIVE.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-Learn**
- **Joblib**
- **Streamlit**

---

## 📂 Project Structure

```text
Titanic-Streamlit-App/
│
├── app.py
├── titanic_model.pkl
├── requirements.txt
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `titanic_model.pkl` | Trained Decision Tree model |
| `requirements.txt` | Required Python packages |
| `README.md` | Project documentation |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Titanic-Streamlit-App.git
```

Move into the project directory:

```bash
cd Titanic-Streamlit-App
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

Run the following command:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deployment

This application can be deployed for free using **Streamlit Community Cloud**.

### Deployment Steps

1. Create a public GitHub repository.
2. Upload:
   - `app.py`
   - `titanic_model.pkl`
   - `requirements.txt`
   - `README.md`
3. Open Streamlit Community Cloud.
4. Sign in using GitHub.
5. Select the GitHub repository.
6. Select the `main` branch.
7. Set the main file to:

```text
app.py
```

8. Click **Deploy**.
9. Wait for the application to build.
10. Open the generated public URL.

---

## 🧪 Example Prediction

Example input:

```text
Passenger Class: 1
Sex: Female
Age: 25
Siblings/Spouses: 0
Parents/Children: 0
Fare: 100
```

The application processes these values and provides a survival prediction.

---

## 📸 Screenshot

Add a screenshot of the working prediction interface here after deployment.

Example:

```text
![Titanic Survival Prediction](prediction_screenshot.png)
```

---

## ⚠️ Limitations

- The Titanic dataset is relatively small.
- Predictions depend on the features provided by the user.
- Filling missing age values with the median may not represent the actual passenger age.
- A Decision Tree can be sensitive to the training data and may overfit.
- This application is intended for educational purposes and should not be considered a real-world prediction system.

---

## 🔐 Privacy

This application does not require or collect personal information. The inputs are used only to demonstrate the machine learning prediction.

---

## 🎓 Learning Outcomes

Through this project, the following concepts were practiced:

- Saving machine learning models with Joblib
- Loading trained models
- Building Streamlit applications
- Creating interactive input controls
- Generating live predictions
- Using model caching
- Managing Python dependencies
- Deploying a machine learning application
- Connecting GitHub with Streamlit Community Cloud

---

## 📜 Disclaimer

This project is created for **educational and demonstration purposes only**. The predictions are based on a historical Titanic dataset and should not be used for real-world decision-making.

---

## 👩‍💻 Author

**Taruni Baisani**

Machine Learning / Data Science Project
