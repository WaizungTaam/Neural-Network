#include <iostream>
#include "../src/vector.hh"
#include "../src/matrix.hh"

int main() {
  Matrix f(2, 3);
  std::cout << "Enter 6 numbers: ";
  std::cin >> f;
  Matrix a(3, 4);
  std::cout << a << std::endl;
  Matrix b(3, 4, 1);
  std::cout << b << std::endl;
  Matrix c = b;
  std::cout << c << std::endl;
  Matrix d = {{1, 2, 3 ,4}, 
              {5, 6, 7, 8}, 
              {9, 10, 11, 12}};
  std::cout << d << std::endl;
  a = d;
  std::cout << a << std::endl;
  b = 3;
  std::cout << b << std::endl;
  b = {{1, 0, 0, 0},
       {0, 1, 0, 0},
       {0, 0, 1, 0}};
  std::cout << b << std::endl;
  std::cout << a.shape()[0] << " " 
            << a.shape()[1] << std::endl;
  Vector u = {1.1, 1.1, 1.1, 1.1};
  Vector v = {2.2, 2.2, 2.2};
  std::cout << b.insert(u, 0, 1) << std::endl;
  std::cout << b.insert(v, 1, 1) << std::endl;  // ***
  std::cout << b.insert(a, 0, 1) << std::endl;
  std::cout << b.insert(a, 1, 1) << std::endl;  // ***
  std::cout << b.remove(0, 1) << std::endl;
  std::cout << b.remove(1, 1) << std::endl;  // ***
  std::cout << b.remove(0, 1, 2) << std::endl;
  std::cout << b.remove(1, 1, 2) << std::endl;  // ***
  std::cout << b.replace(u, 0, 1) << std::endl;
  std::cout << b.replace(v, 1, 1) << std::endl;
  std::cout << b.replace(a, 0, 1) << std::endl;
  std::cout << b.replace(a, 1, 1) << std::endl;
  std::cout << a.T() << std::endl;
  std::cout << a.reshape(6, 2) << std::endl;
  std::cout << a.sum() << std::endl;
  std::cout << a.cross(b) << std::endl;
  d = {{1, 3, 5, 7},
       {2, 4, 6, 8},
       {3, 5, 7, 9}};
  std::cout << a + d << std::endl;
  std::cout << a + 2 << std::endl;
  std::cout << 2 + a << std::endl;
  std::cout << a - d << std::endl;
  std::cout << a - 2 << std::endl;
  std::cout << 2 - a << std::endl;
  std::cout << a * (d.T()) << std::endl;
  std::cout << a * 2 << std::endl;
  std::cout << 2 * a << std::endl;
  std::cout << a / d << std::endl;
  std::cout << a / 2 << std::endl;
  std::cout << 2 / a << std::endl;
  d += 1;
  std::cout << d << std::endl;
  d -= 1;
  std::cout << d << std::endl;
  d *= 2;
  std::cout << d << std::endl;
  d /= 2;
  std::cout << d << std::endl;
  Matrix e = d;
  std::cout << (d == e) << std::endl;
  std::cout << (d != e) << std::endl;
  std::cout << a[1][2] << std::endl;
  std::cout << a[2][3] << std::endl;
  a[0][0] = 999.9;
  std::cout << a << std::endl;
  std::cout << a(1, 2, 1, 3) << std::endl;
  std::cout << a(0, 2, 0, 1) << std::endl;
  Matrix g = u;
  std::cout << g << std::endl;
  Matrix h(1, 3);
  h = v;
  std::cout << h << std::endl;
  std::cout << a * u << std::endl;
  std::cout << a[1] << std::endl;
  std::cout << a(1) << std::endl;
  std::cout << (a < d) << std::endl;
  std::cout << (a < 5) << std::endl;
  std::cout << (5 < a) << std::endl;
  std::cout << (a <= d) << std::endl;
  std::cout << (a <= 5) << std::endl;
  std::cout << (5 <= a) << std::endl;
  Matrix i;
  std::cout << i << std::endl;
  std::cout << a.max(0) << std::endl;
  std::cout << a.min(1) << std::endl;
  Matrix j(3, 4, "uniform", -1.0, 1.0);
  std::cout << j << std::endl;
  Matrix k(3, 4, "normal", 0.0, 1.0);
  std::cout << k << std::endl;
  Matrix l = {{2}, {3}, {4}};
  std::cout << a.shape()[0] << " " << a.shape()[1] << std::endl;
  std::cout << l.shape()[0] << " " << l.shape()[1] << std::endl;
  std::cout << a + l << std::endl;
  std::cout << a.shuffle() << std::endl;
  return 0;
}