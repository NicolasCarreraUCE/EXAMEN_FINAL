#include <iostream>
#include <vector>
#include <fmt/core.h>
#include <mpi.h>

#define M 32
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0 && M % size != 0) {
        fmt::println("M = {} no es múltiplo por el número de RANKs = {}", M, size);
        MPI_Finalize();
        return 0;
    }

    std::vector<int> A(M * M, 0);
    std::vector<int> B(M, 0);

    // Inicializa la Matriz A y B
    if (rank == 0) {
        for (int i = 0; i < M * M; i++) {
            A[i] = i + 1;
        }
        for (int i = 0; i < M; i++) {
            B[i] = i + 1;
        }

        for (int i = 1; i < size; i++) {
            //  Comunicación punto a punto: Envío de datos
            MPI_Send(A.data(), M * M, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(B.data(), M, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        //  Comunicación punto a punto: Recibimiento de datos
        MPI_Recv(A.data(), M * M, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B.data(), M, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int block_size = M * M / size;
    std::vector<int> local_A(block_size);

    // Comunicación colectiva: Distribuye los datos entre los procesos
    MPI_Scatter(A.data(), block_size, MPI_INT, local_A.data(), block_size, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> local_sum(M / size, 0);
    for (int i = 0; i < M / size; i++) {
        for (int j = 0; j < M; j++) {
            local_sum[i] += local_A[i * M + j] * B[j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> AB(M);
    // Comunicación colectiva: Junta los datos de los demas procesos al rank 0
    MPI_Gather(&local_sum[0], M / size, MPI_INT, &AB[0], M / size, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for( int i = 0; i < M ; i++) {
            fmt::println( "[{}]", AB[i]);
        }
    }

    MPI_Finalize();
    return 0;
}
