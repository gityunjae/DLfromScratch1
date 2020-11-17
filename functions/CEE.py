def cross_entropy_error(y, t):
  delta = 1e-7 # 로그의 결과가 -inf가 나오는 것을 방지하기 위해 아주 작은 값을 더해줌
  return -np.sum(t*np.log(y+delta))

# 미니 배치에서 교차엔트로피 오차를 구하는 방법
def cross_entropy_error2(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
    
  batch_size = y.shape[0]
  return -np.sum(t*np.log(y+1e-7)) / batch_size

# 미니 배치에서 정답이 숫자 레이블로 주어졌을 때
def cross_entropy_error3(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
    
  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
