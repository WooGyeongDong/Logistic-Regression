PCA, Likelihood, LogisticRegression
=================

# 1.PCA(Principal Component Analysis) 전처리
- PCA는 가장 큰 분산을 가지는 부분공간을 유지하는 최적 직교 변환이다. 따라서 PCA는 변수의 크기 조정에 민감하다. 
- 만약 같은 분산을 갖는 두 개의 변수가 있고 이 둘에 양의 상관관계가 존재한다면, PCA는 45도 회전을 수반할 것이고 이 때 주성분에 대한 두 변수 각각의 가중치는 동일할 것이다. 그러나 동일한 상황에서 첫 번째 변수의 값들이 100배가 되면 첫 번째 주성분은 이 첫 번째 변수와 거의 같은 방향이 되고 두 번째 변수에는 매우 적은 영향만을 받게 된다. 반면 두 번째 성분은 거의 두 번째의 원래 변수들과 정렬된다. 
- 따라서 변수들이 서로 다른 단위(기온, 질량 처럼)를 가지면 PCA는 임의적인 분석방법이 될 수 있다. 이를 해결하는 방법 중 하나는, 각각의 변수들을 표준화시켜 모두 단위 분산을 갖게 하는 것이다. 그러나 이 방법은 신호공간의 모든 차원에서 값들의 변동을 단위 분산으로 압축(또는 팽창)시킨다.

# 2. 딥러닝의 Likelihood관점에서의 해석
- 목적함수
$$\hat y=f(f(XW^{(1)} + 1b^{(1)})W^{(2)}+1b^{(2)}),\;\;\;\;f(x):elemental-wise\;nonlinear\;activation$$
$$Loss(\hat y,y)=MSE(\hat y,y)=\frac{1}{n}(y-\hat y)^T(y-\hat y)$$

- 자료의 오차가 정규분포를 따른다고 가정하면 가능도함수는 다음과 같다.
$$L(\beta;y)=\Pi\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{1}{2\sigma^2}(y_i-\hat y_i)^2)$$
$$=\frac{1}{\sqrt{2\pi}\sigma}^nexp(-\frac{1}{2\sigma^2}\sum(y_i-\hat y_i)^2)$$
$$\log L(\beta;y)=-n\log \sqrt{2\pi}\sigma-\frac{1}{2\sigma^2}\sum(y_i-\hat y_i)^2$$

- 따라서 로그 가능도함수를 최대화 하는 것은 MSE를 최소화 하는 것과 동치이다.
$$\underset{\beta}{max}\log L(\beta;y)\Leftrightarrow \underset{\beta}{min}\sum(y_i-\hat y_i)^2\Leftrightarrow \underset{\beta}{min}\;MSE$$


# 3. Logistic Regression
- Notation
- $$X\in \mathbb{R}^{n\times (k+1)}:Train\;Data$$
$$X= \begin{pmatrix}
 x_1\\
 x_2\\
 \vdots\\
x_n\end{pmatrix},\;\;\;\;n=\#\;of\;sample$$
$$x_i=\begin{pmatrix}
1 & x_{i1} & x_{i2} & \cdots & x_{ik} \\
\end{pmatrix},\;\;\;\;k=\#\;of\;factor$$
$$\beta=\begin{pmatrix}
\beta_0 & \beta_1 & \cdots & \beta_k \\
\end{pmatrix}^T\in\mathbb{R}^{k+1}:Parameter$$
$$f(x)=\frac{exp(x)}{1+exp(x)}:Sigmoid\;function$$
$$\hat y=f(X\beta)\in \mathbb{R}^{n}:Predict\;value$$
$$y\in \{0,1\}^n:True\;value$$

- Cross Entropy
$$L(\beta;y)=-\sum [y_i\log f(x_i\beta)+(1-y_i)\log (1-f(x_i\beta))]$$
$$=-\sum [y_i(\log exp(x_i\beta)-\log (1+exp(x_i\beta))-(1-y_i)\log (1+exp(x_i\beta))]$$
$$=-\sum [y_ix_i\beta-\log(1+exp(x_i\beta))]$$

- Minimize Loss
$$\underset{\beta}{min}L(\beta;y)\Rightarrow\frac{\partial L(\beta;y)}{\partial \beta}=0$$
$$\frac{\partial L(\beta;y)}{\partial \beta_j}=-\sum [y_ix_{ij}-\frac{exp(x_i\beta)}{1+exp(x_i\beta)}x_{ij}]$$
$$=\sum (\hat y_i-y_i)x_{ij}=x_i^T(\hat y_i-y_i)$$
$$\frac{\partial L(\beta;y)}{\partial \beta}=X^T(\hat y_i-y_i)$$

### Data Generation
```
import numpy as np
#Set beta
beta = np.array([0.4, -0.2, 0.3, 0.5, 0.6, -0.7])

# Generate X
n = 500
np.random.seed(seed=0)
X = np.random.normal(0,3, size=(n,len(beta)-1))
const = np.ones(n)
X = np.hstack((const.reshape(n,1),X))

# Generate y
mu = X@beta
prob = 1/(np.exp(-mu)+1)
y = prob.round()
```
### Logistic Regression
```
#Gradient Decent Algorithm
class LogisticRegression:
    def __init__(self, learning_rate, num_iter) -> None:
        self.learnig_rate = learning_rate
        self.num_iter = num_iter

    def gradient(self, x, y, b):
        return x.T@(-y+1/(np.exp(-x@b)+1))

    def loss(self, x, y, b):
        cost = (-y*np.log(1/(1+np.exp(-x@b)))-(1-y)*np.log(1-1/(1+np.exp(-x@b)))).mean()
        return cost 
    
    def fit(self, X, y):
        
        self.beta_hat = np.random.normal(0,1, size=(len(beta)))
        
        self.total_loss = []
        for i in range(self.num_iter):
            self.beta_hat -= self.learnig_rate * self.gradient(X, y, self.beta_hat)
            self.total_loss.append(self.loss(X, y, self.beta_hat))
            
            if i % 1000 == 0:
                print(self.beta_hat, self.loss(X, y, self.beta_hat))
        
    def predict(self, x):
        predict_prob = 1/(1+np.exp(-x@self.beta_hat))
        return predict_prob.round()
        
model = LogisticRegression(0.0001, 10000)
model.fit(X,y)

print("Learned beta: {} \nTrue beta: {})".format(np.round(model.beta_hat,4), beta))
np.mean(model.predict(X) == y)
```
### 결과
```
Learned beta: [ 0.4286 -0.2562  0.3251  0.485   0.6253 -0.7556] 
True beta: [ 0.4 -0.2  0.3  0.5  0.6 -0.7])
```
- Iteration에 따른 Loss 변화
![output10.26.png](https://www.dropbox.com/scl/fi/vtn5hl9c4d1m8nhmn34ra/output10.26.png?rlkey=wh2qhkivadol656g9nbcw2fdm&dl=0&raw=1)
- Test Data ACC
0.846