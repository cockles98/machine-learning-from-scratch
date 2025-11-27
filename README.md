# 🧠 Machine Learning from Scratch (NumPy Implementation)

> **"What I cannot create, I do not understand." — Richard Feynman**

Este repositório contém implementações puras em Python (via `NumPy`) de algoritmos fundamentais de Machine Learning e Deep Learning. O objetivo deste projeto não é substituir frameworks como PyTorch ou Scikit-Learn, mas sim desconstruir a "caixa preta" para dominar a matemática, a otimização e a vetorização que sustentam a inteligência artificial moderna.

---

## 🎯 Objetivo e Filosofia
Em finanças quantitativas e pesquisa de IA, entender a derivada de uma função de custo ou a estabilidade numérica de uma inversão de matriz é crucial. Este projeto foca em:

1.  **Vetorização Total:** Eliminação de loops `for` desnecessários usando operações matriciais (Broadcasting/Dot Products) para máxima performance.
2.  **Rigor Matemático:** Implementação fiel das equações de gradiente descendente, *backpropagation* e otimização convexa.
3.  **Modularidade:** Arquitetura orientada a objetos seguindo a API padrão `fit(X, y)` / `predict(X)`.

---

## 🛠️ Algoritmos Implementados

### Supervised Learning
* **Linear Regression:**
    * Métodos: OLS (Ordinary Least Squares) via Equação Normal e Gradient Descent.
    * Regularização: Ridge (L2) e Lasso (L1).
* **Logistic Regression:**
    * Otimização via Gradient Descent com função de perda Log-Loss (Binary Cross-Entropy).
* **Decision Trees (CART):**
    * Cálculo recursivo de *Information Gain* e *Gini Impurity*.
* **K-Nearest Neighbors (KNN):**
    * Cálculo eficiente de distâncias (Euclidiana, Manhattan) vetorizadas.
* **Support Vector Machines (SVM):**
    * Otimização usando Hinge Loss e Kernel Trick (RBF/Linear).

### Unsupervised Learning
* **K-Means Clustering:**
    * Algoritmo de Expectation-Maximization com inicialização aleatória.
* **Principal Component Analysis (PCA):**
    * Decomposição de autovalores/autovetores (Eigendecomposition) da matriz de covariância.

### Deep Learning
* **Multilayer Perceptron (MLP):**
    * *Forward Pass:* Ativações (Sigmoid, ReLU, Tanh, Softmax).
    * *Backward Pass:* Implementação manual da regra da cadeia para cálculo de gradientes.
    * *Otimizadores:* SGD, Momentum, Adam.

---

## ⚡ Exemplo de Código: Regressão Linear Vetorizada

Um exemplo de como a matemática é traduzida diretamente para operações matriciais eficientes:

```python
class LinearRegression:
    def fit(self, X, y):
        # Adiciona termo de bias (intercept)
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Closed-form solution (Normal Equation): theta = (X.T * X)^-1 * X.T * y
        # Utiliza pseudo-inversa para estabilidade numérica
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.theta
````

-----

## 📐 O "Motor" Matemático

Por trás do código, o foco está na derivação correta dos gradientes. Exemplo para a Regressão Logística:

A função de custo (Log-Loss):

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(h_\theta(x^{(i)})) + (1 - y^{(i)})\log(1 - h_\theta(x^{(i)}))]
$$

O gradiente vetorizado para atualização dos pesos:

$$
\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^T (h_\theta(X) - y)
$$

-----

## 🚀 Como Executar

O projeto utiliza `poetry` (ou `pip`) para gerenciamento de dependências.

```bash
# Clone o repositório
git clone [https://github.com/cockles98/machine-learning-from-scratch.git](https://github.com/cockles98/machine-learning-from-scratch.git)

# Instale as dependências (apenas numpy e matplotlib para visualização)
pip install -r requirements.txt

# Execute os notebooks de exemplo
jupyter notebook notebooks/Linear_Regression_Demo.ipynb
```

-----

## 📊 Comparação de Performance

Benchmarks realizados contra a implementação padrão do `scikit-learn` em datasets sintéticos (100k samples):

| Algoritmo | Tempo (My Implement.) | Tempo (Scikit-Learn) | Acurácia Relativa |
| :--- | :--- | :--- | :--- |
| Linear Reg. (Normal Eq) | 0.04s | 0.02s | 100% |
| Logistic Reg. (GD) | 0.15s | 0.09s | 98.5% |
| K-Means | 0.32s | 0.12s | 99.0% |

*Nota: O Scikit-learn utiliza rotinas em Cython/C para otimização extrema, mas nossa implementação vetorizada em NumPy mantém performance competitiva para fins educacionais.*
