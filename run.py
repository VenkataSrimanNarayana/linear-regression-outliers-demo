import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Create a streamlit app
st.title("Linear Regression")
st.write(
    "This app shows the fit of a Linear regression model with varying fraction of outliers and Regularization type."
)

# Value for meu for lasso or ridge
meu = None

# The sidebar contains the sliders
with st.sidebar:
    st.write("#### Options:")

    # Create a slider for Number of data points(including outliers)
    n = st.slider("Number of data points", 10, 100, 10)
    
    # Create a slider for Fraction of outliers
    alpha = st.slider("Fraction of outliers", 0.0, 1.0, 0.0)

    # Create a slider for degree
    degree = st.slider("Degree", 1, 5, 2)

    # Option to select the model type
    model_type = st.radio(
    "Choose the Regularization type",
    ('None', 'Ridge', 'Lasso'))

    if (model_type != 'None'):
        meu = st.slider("alpha", 0.0, 10.0, 1.0)

# True function
def f(x):
    return 10 + x**2

# Generating the data
np.random.seed(0)
X = np.linspace(1, 10, n)
e = np.random.randn(*X.shape) # Noise
y = f(X) + 4 * e

# Inroducing outliers
indices = np.random.choice(np.arange(0, X.size), int(alpha * X.size), replace=False)
y[indices] = np.random.uniform(0, 90, size=indices.size)

# Creating and training model
model_class = LinearRegression
params = []
if (model_type == "Ridge"):
    params.append(meu)
    model_class = Ridge
elif (model_type == "Lasso"):
    params.append(meu)
    model_class = Lasso
model = make_pipeline(PolynomialFeatures(degree), model_class(*params))
model.fit(X.reshape(-1, 1), y)

# Plotting the model
x_plot = np.arange(1, 10, 0.01)
y_plot = model.predict(x_plot.reshape(-1, 1))
fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(x_plot, y_plot, label="Predicted", color="r")
ax.plot(x_plot, f(x_plot), label="True", color="k", linestyle="--")
# Function add a legend  
plt.legend()
ax.set_xlabel("X")
ax.set_ylabel("y")
st.pyplot(fig)

st.write("---")
st.write(f"**Maximum coefficient:** {max(model.steps[1][1].coef_)}")
st.write(f"**R2 score:** {model.score(X.reshape(-1, 1), y)}")
