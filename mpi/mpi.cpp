#include <mpi.h>
#include <mkl.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace std;

// скорректировать диагональ матрицы
void correct_diag(int rank, int row, int col, double* matrix) {
    int row_shift = col * rank;
    double* matrix_shift = matrix + row_shift * col;
    for (int i = 0; i < col; ++i) {
        matrix_shift[i * col + i] += (double)(row_shift + i + 1) / (10 * row);
    }
    return;
}
// глобальное скалярное произведение (x, y)
double global_ddot(int col, const double* x, const double* y) {
    double local_ddot = cblas_ddot(col, x, 1, y, 1);
    double dot_gl;
    MPI_Allreduce(&local_ddot, &dot_gl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return dot_gl;
}

// y = matrix * x
void global_dgemv(int rank, int size, int row, int col, const double* matrix, const double* x, double* y) {
    vector<double> part_sums(row);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, row, col, 1, matrix, col, x, 1, 0.0, part_sums.data(), 1);
    vector<int> recv_counts(size, col); 
    MPI_Reduce_scatter(part_sums.data(), y, recv_counts.data(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return;
}

// y = matrix * x
//void global_dgemv(int rank, int size, int row, int col, const double* matrix, const double* x, double* y) {
//    vector<double> part_sums(row);
//    vector<MPI_Request> requests(size);
//    for (int target_rank = 0; target_rank < size; ++target_rank) {
//        cblas_dgemv(CblasRowMajor, CblasNoTrans, col, col, 1, matrix + target_rank * col * col, col, x, 1, 0.0, part_sums.data() + target_rank * col, 1);
//        MPI_Ireduce(part_sums.data() + target_rank * col, (target_rank == rank) ? y : nullptr, col, MPI_DOUBLE, MPI_SUM, target_rank, MPI_COMM_WORLD, &requests[target_rank]);
//    }
//    MPI_Waitall(size, requests.data(), MPI_STATUSES_IGNORE);
//    return;
//}

// решение СЛАУ: matrix * x = f методом CG
void CG(int rank, int size, int row, int col, double eps, const double* matrix, const double* f, double* x) {
    int iter = 0;
    double alpha, beta, rr;
    eps *= sqrt(global_ddot(col, f, f));
    vector<double> r(col), p(col), Ap(col);
    memcpy(r.data(), f, col * sizeof(double));
    memcpy(p.data(), r.data(), col * sizeof(double));
    rr = global_ddot(col, r.data(), r.data());
    while (sqrt(rr) > eps) {
        global_dgemv(rank, size, row, col, matrix, p.data(), Ap.data());
        alpha = rr / global_ddot(col, p.data(), Ap.data());
        cblas_daxpy(col, alpha, p.data(), 1, x, 1);
        cblas_daxpy(col, -alpha, Ap.data(), 1, r.data(), 1);
        beta = rr;
        rr = global_ddot(col, r.data(), r.data());
        beta = rr / beta;
        cblas_dscal(col, beta, p.data(), 1);
        cblas_daxpy(col, 1.0, r.data(), 1, p.data(), 1);
        iter++;
        /*if (iter % 1 == 0) {
            if (rank == 0) cout << iter << ' ' << eps << ' ' << sqrt(rr) << endl;
        }*/
    }
    //cout << iter << ' ' << sqrt(rr) << ' ';
    return;
}
//
void procedure(int argc, char** argv, int rank, int size, int row) {
    double eps = pow(10, -8);
    int col = row / size;
    vector<double> matrix(row * col, 1.0);
    correct_diag(rank, row, col, matrix.data());
    vector<double> f(col, 1.0);
    vector<double> x(col, 0.0);
    double start = MPI_Wtime();
    CG(rank, size, row, col, eps, matrix.data(), f.data(), x.data());
    double end = MPI_Wtime();
    if (rank == 0) cout << row << ' ' << size << ' ' << end - start << ' ' << endl;
    return;
}

// mpiexec -n 4 mpi.exe 100
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int row = stoi(argv[1]);
    procedure(argc, argv, rank, size, row);

    MPI_Finalize();

    return 0;
}