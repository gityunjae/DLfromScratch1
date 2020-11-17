def cross_entropy_error(y, t):
  delta = 1e-7 # 로그의 결과가 -inf가 나오는 것을 방지하기 위해 아주 작은 값을 더해줌
  return -np.sum(t*np.log(y+delta))
