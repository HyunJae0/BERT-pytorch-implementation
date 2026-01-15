import torch
import math

## 1. torch.polar(abs, angle)
"""
torch.polar(abs, angle) 함수는 
극좌표계 형식의 데이터(abs, angle)를 입력받아, 직교좌표계 형식의 복소수(즉 z=x+yi⋅) 텐서로 변환하는 데 사용된다.

다음 두 가지 정보를 입력으로 받는다:
- abs: 원점으로부터의 거리 r
- angle: 기준축으로부터의 회전 각도 θ

이 정보를 이용하여 아래 공식을 통해 복소수 z = a + b⋅i를 생성한다.
out = abs⋅cos(angle)+abs⋅sin(angle)⋅i 
"""

"""
먼저, 극좌표와 직교좌표 사이의 관계를 생각해 보면,
평면상의 한 점 p=(x, y)에 대해 원점(o)에서 p까지의 거리를 r이라 하고, x축에서 op까지의 각을 θ라고 할때, 점 p를 극좌표 (r, θ)로 나타낼 수 있다. 
- 예: 직교좌표 (1,1) -> 극좌표(sqrt(2), pi/4)

직교좌표의 (x, y)와 극좌표(r, θ)의 관계는 삼각함수를 통해 알 수 있다. 
# 삼각함수 그림 참고: https://blog.naver.com/semomath/222939867201 
삼각함수 정의에 따라 cosθ = x/r, sinθ = y/r이며, 이는 다시 x = r⋅cosθ, y = r⋅sinθ로 나타낼 수 있다.
즉, 직교좌표의 x와 y를 극좌표의 구성요소인 r과 θ로 표현할 때는 x = r⋅cosθ, y = r⋅sinθ로 표현한다. 
또한, x^2+y^2=r^2이란 것도 알 수 있다. 

x = r⋅cosθ, y = r⋅sinθ로 임의의 복소수 z = x+y⋅i를 다음과 같이 표현할 수 있다. 
z = r⋅cosθ + r⋅sinθ⋅i -> 이것이 torch.polar가 계산하는 식이다. 

여기에 오일러 공식 e^iθ = cosθ + i⋅sinθ를 복소수 z에 곱하면 복소수 z가 2차원 평면에서 θ만큼 회전한다.
RoPE는 이 회전 성질을 이용해서 positional encoding을 구현한다. 토큰의 위치 정보를 embedding 공간에서 회전으로 인코딩한다.

오일러의 공식으로 복소수의 극좌표 표현을 다음과 같이 지수로 나타낼 수 있다.
e^iθ = cosθ + i⋅sinθ -> r⋅e^iθ = r⋅cosθ + r⋅sinθ⋅i = z => z = r⋅e^iθ 
z = r⋅e^iθ는 z의 크기가 r이고, z의 각도가 θ라는 것을 의미한다. 

복소수에 오일러 공식을 곱하면 다음과 같다.
z' = z x e^iθ
= (x+y⋅i) x (cosθ + i⋅sinθ)
= (x⋅cosθ - y⋅sinθ) + (x⋅sinθ + y⋅cosθ)⋅i
가 되며, 첫 번째 항이 실수부이고 두 번째 항이 허수부이다. 
실수부 (x⋅cosθ - y⋅sinθ)를 x', 허수부 (x⋅sinθ + y⋅cosθ)⋅i를 y'으로 둔다면, 다음과 같이 나타낼 수 있다.
(x')   (cosθ -sinθ)(x)
    = 
(y')   (sinθ  cosθ)(y) # 여기서 2차원 회전 행렬이 유도되는 것을 볼 수 있다. 

여기서 z가 z = r e^{iφ}라고 한다면, 복소수 r e^{iφ}에 e^iθ가 곱해져서 벡터의 크기 r은 유지되고 각도만 변하게 된다.
정확하게 말하면, 각도 φ를 가진 벡터를 θ만큼 회전시키는 것이다. 
그래서 RoPE에서는 토큰(정확하게는 임베딩 벡터)이 가진 본래의 의미 정보를 훼손하지 않는다. 벡터의 크기가 유지되기 때문이다.
"""

"""
torch.polar(abs, angle) 함수를 다시 보면, 
out = r⋅cos(θ)+r⋅sin(θ)⋅i = r(cos(θ)+sin(θ)⋅i) = r⋅e^iθ = z 라는 것을 알 수 있다.
즉, 오일러 공식을 활용하여 극좌표(r, θ)를 복소수z = r⋅e^iθ = r⋅cosθ + r⋅sinθ⋅i로 변환하는 것이다.

torch.polar의 입력 abs와 angle은 동일한 dtype을 가져야 하며, torch.polar의 output은 해당 dtype에 대응하는 복소수 dtype을 가진다.
예를 들어 float32면 output은 complex64, float64면 complex128
"""

r = torch.tensor([1, 2], dtype=torch.float64)
theta = torch.tensor([math.pi / 2, 5*math.pi / 4], dtype=torch.float64)
z = torch.polar(r, theta)
print(z)
"""
첫 번째 요소는 r = 1, θ = pi/2 (90°)
실수부: 1⋅cosθ = cos90° = 0
허수부: 1⋅sinθ = sin90° = 1
-> 결과: 0 + 1⋅i

두 번째 요소는 r = 2, θ = 5*pi/4 (225°)
실수부: 2⋅cos225° = 2⋅(-sqrt(2)/2) = -sqrt(2)
허수부: 2⋅sin225° = 2⋅(-sqrt(2)/2) = -sqrt(2)
-> 결과 -sqrt(2) + -sqrt(2)⋅i

이렇게 크기와 각도를 입력으로 받아 complex tensor로 변환한다.
"""

## 2. torch.view_as_complex()와 torch.view_as_real()
"""
torch.view_as_complex(input tensor)는 텐서의 데이터를 복소수로 변환하는 함수이다.
이름 그대로, 텐서의 데이터를 복사하지 않고 메모리를 공유하는 view(뷰) 형태로 변환한다. 

직교좌표(x, y) 형태로 짝지어 있는 텐서의 데이터를 복소수 텐서로 바꿔주는 역할을 한다. 단, float32 또는 float64만 가능하다. 
예를 들어 [N, 2] 크기의 실수 텐서(real tensor)가 있다면, 이 함수를 통해 [N] 크기의 complex tensor로 변환할 수 있다.
그래서 마지막 차원 크기는 무조건 2(실수부, 허수부)여야 한다. 
"""

x = torch.randn(4, 2)
x_complex = torch.view_as_complex(x)
print(x_complex)

x_transposed = x.transpose(0, 1)
# print(torch.view_as_complex(x_transposed)) 에러 발생
"""
이 함수는 실수부와 허수부가 메모리 상에서 딱 붙어 있어야 한다. 
텐서가 contiguous하지 않으면 에러가 발생한다. 
"""

"""
torch.view_as_real(input tensor)는 torch.view_as_complex(input tensor)의 정반대 역할을 하는 함수이다.
complex tensor를 실수부와 허수부로 분리해 마지막 차원이 2(실수부와 허수부)인 real tensor로 변환해준다. 
"""
x_real = torch.view_as_real(x_complex)
print(x_real)
print(x.shape); print(x_complex.shape);print(x_real.shape)

## 3. torch.outer()
"""
torch.outer(vec1, vec2)는 1D vector vec1과 1D vector vec2의 outer product를 계산하는 함수이다. 
vec1의 size가 n이고 vec2의 size가 m이라면, torch.outer(vec1, vec2)의 output은 n x m size의 tensor이다.
"""
v1 = torch.arange(1., 5.)
v2 = torch.arange(1., 4.)
print(v1.shape, v2.shape)
print(torch.outer(v1, v2))
print(torch.einsum("i, j -> ij", v1, v2))

## 4. torch.type_as()
"""
Tensor.type_as(tensor)로 사용한다. 
a.type_as(b)라면, tensor a의 dtype을 tensor b의 dtype으로 만들어주는 함수이다. 

type_as()는 in-place 연산이 아니기 때문에 다음과 같이 사용해야 한다. 
c = a.type_as(b)로 하면, input tensor a가 어떤 dtype이든 b의 dtype으로 바꿔준다.
그리고 dtype뿐만 아니라 device도 함께 변경된다. 
"""
a = torch.arange(1, 5, dtype=torch.float64, device=torch.device('cpu'))
b = torch.arange(1, 10, dtype=torch.float32, device=torch.device('cuda'))
c = a.type_as(b)
print(c)
print(a.dtype, b.dtype, c.dtype)
print(a.device, b.device, c.device)