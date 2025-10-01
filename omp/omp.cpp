#include <omp.h>
#include <vector>
#include <iostream>
using namespace std;

// скорректировать диагональ матрицы
void correct_diag(int n, double* matrix) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        matrix[i * n + i] += (double)(i + 1) / (10 * n);
    }
    return;
}
// скалярное произведение (x, y)
double ddot(int n, const double* x, const double* y) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}
// y = matrix * x
void dgemv(int n, int m, const double* matrix, const double* x, double* y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double local_sum = 0.0;
        const double* row = matrix + i * m;
        for (int j = 0; j < m; ++j) {
            local_sum += row[j] * x[j];
        }
        y[i] = local_sum;
    }
    return;
}
// y += alpha * x
void daxpy(int n, double alpha, const double* x, double* y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
    return;
}
// y *= alpha
void dscal(int n, double alpha, double* x) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
    return;
}
// решение СЛАУ: matrix * y = x методом CG
void CG(int n, double eps, const double* matrix, const double* f, double* x) {
    int iter = 0;
    double alpha, beta, rr;
    eps *= sqrt(ddot(n, f, f));
    vector<double> r(n), p(n), Ap(n);

    memcpy(r.data(), f, n * sizeof(double));
    memcpy(p.data(), r.data(), n * sizeof(double));
    rr = ddot(n, r.data(), r.data());
    while (sqrt(rr) > eps) {
        dgemv(n, n, matrix, p.data(), Ap.data());
        alpha = rr / ddot(n, p.data(), Ap.data());
        daxpy(n, alpha, p.data(), x);
        daxpy(n, -alpha, Ap.data(), r.data());
        beta = rr;
        rr = ddot(n, r.data(), r.data());
        beta = rr / beta;
        dscal(n, beta, p.data());
        daxpy(n, 1.0, r.data(), p.data());
        iter++;
        //if (iter % 10 == 0) cout << iter << ' ' << eps << ' ' << sqrt(rr) << endl;
    }
    return;
}
// 
void procedure(int n) {
    double eps = pow(10, -8);
    vector<double> matrix(n * n, 1.0);
    correct_diag(n, matrix.data());
    vector<double> f(n, 1.0);
    vector<double> x(n);
    double start = omp_get_wtime();
    CG(n, eps, matrix.data(), f.data(), x.data());
    double end = omp_get_wtime();
    cout << end - start << ' ' << endl;
    return;
}

int main() {
    for (int n = 128; n < 8193; n *= 2) {
        for (int t = 1; t < 17; t *= 2) {
            omp_set_num_threads(t);
            cout << n << ' ' << t << ' ';
            procedure(n);
        }
        cout << endl;
    }
    return 0;
}
