
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const float interval_begin = 0.0;
const float interval_end = 1.0;

const float temperature_begin = 0.0;
const float temperature_end = 0.0;

const int tag_init = 1;
const int tag_computing = 2;
const int tag_gathering = 3;

float initial_temperature(float x) {
    return 0.5;
}

void print_usage_and_die() {
    fprintf(stderr, "Usage: solve <number of points> <T, time to integrate> <time step>\n");
    exit(EXIT_FAILURE);
}

float compute_next_step(float time_step, float x_step, float y1, float y2, float y3) {
    return y2 + time_step / x_step * (y1 - 2 * y2 + y3);
}

void parse_arguments_or_die(int argc, char *argv[], int *points_out, 
                         float *time_to_integrate_out, float *time_step_out) {
    if (argc != 4)
        print_usage_and_die();
    *points_out = atoi(argv[1]);
    *time_to_integrate_out = atof(argv[2]);
    *time_step_out = atof(argv[3]);
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int points;
    float time_to_integrate;
    float time_step;
    parse_arguments_or_die(argc, argv, &points, &time_to_integrate, &time_step);

    int processes;
    int my_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (points < 2) {
        if (my_rank == 0) {
            MPI_Finalize();
            fprintf(stderr, "Number of points should be at least 2");
            exit(EXIT_FAILURE);
        }
    }

    if (processes > points) {
        if (my_rank == 0) {
            MPI_Finalize();
            fprintf(stderr, "Number of processes should be at least the number of points\n");
            exit(EXIT_FAILURE);
        }
    }

    // 1. Master process fills the buffer with initial data
    float *ys = NULL;
    if (my_rank == 0) {
        ys = (float *) calloc(points, sizeof(float));
        for (int i = 1; i < points - 1; i++) {
            ys[i] = initial_temperature(interval_begin + i * (interval_end - interval_begin) / points);
        }
        ys[0] = temperature_begin;
        ys[points - 1] = temperature_end;
    } 

    // 2. Processes determine which the subintervals they are responsible for
    int *points_begin = (int *) calloc(processes, sizeof(int));
    int *points_end = (int *) calloc(processes, sizeof(int));
    int *points_num = (int *) calloc(processes, sizeof(int));
    
    for (int rank = 0; rank < processes; rank++) {
        int cur_points_num = points / processes;
        int points_left = points % processes;
        if (rank < points_left) 
            cur_points_num += 1;

        int cur_points_begin, cur_points_end;
        if (rank < points_left) {
            cur_points_begin = rank * cur_points_num;
            cur_points_end = cur_points_begin + cur_points_num - 1;
        } else {
            cur_points_begin = (cur_points_num + 1) * points_left 
                + (rank - points_left) * cur_points_num;
            cur_points_end = cur_points_begin + cur_points_num - 1;
        }

        if (rank > 0) {
            cur_points_begin -= 1;
            cur_points_num += 1;
        }
        if (rank < processes - 1) {
            cur_points_end += 1;
            cur_points_num += 1;
        }

        points_begin[rank] = cur_points_begin;
        points_end[rank] = cur_points_end;
        points_num[rank] = cur_points_num;
    }

    int my_points_num = points_num[my_rank];
    int my_points_begin = points_begin[my_rank];
    int my_points_end = points_end[my_rank];

#ifndef DEBUG
    fprintf(stderr, "Process %d: (%d, %d), %d\n", my_rank, points_begin[my_rank], 
           points_end[my_rank], points_num[my_rank]);
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    // 3. Master process distributes the initial values among the processes
    float *buf1 = (float *) calloc(my_points_num, sizeof(float));
    float *buf2 = (float *) calloc(my_points_num, sizeof(float));
    float *my_ys = buf1;
    float *my_ys_temp = buf2;
    
    if (my_rank == 0) {
        for (int rank = 1; rank < processes; rank++) {
            MPI_Send(ys + points_begin[rank], points_num[rank], MPI_FLOAT, 
                     rank, tag_init, MPI_COMM_WORLD);  
        }
        
        for (int i = my_points_begin; i < my_points_end; i++) 
            my_ys[i] = ys[i];

#ifndef DEBUG
        fprintf(stderr, "Process %d received %d points\n", 0, my_points_num);
#endif
    } else {
        MPI_Recv(my_ys, my_points_num, MPI_FLOAT, 0, tag_init, 
                 MPI_COMM_WORLD, NULL);

#ifndef DEBUG
        fprintf(stderr, "Process %d received %d points\n", my_rank, my_points_num);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 4. Processes start computing the solution
    int iterations_num = time_to_integrate / time_step;
    float x_step = (interval_end - interval_begin) / (points - 1);
    for (int iteration = 0; iteration < iterations_num; iteration++) {
       
        int first_index_to_compute = 1;
        int last_index_to_compute = my_points_num - 2;

        for (int i = first_index_to_compute; i <= last_index_to_compute; i++) {
            my_ys_temp[i] = compute_next_step(time_step, x_step, my_ys[i - 1], my_ys[i], my_ys[i + 1]);
        }

        // Pass the border points to neighbor processes
        if (my_rank % 2) {
            if (my_rank < processes - 1) {
    //            fprintf(stderr, "\tProcess %d -> Process %d\n", my_rank, my_rank + 1);
                MPI_Send(&my_ys_temp[last_index_to_compute], 1, MPI_FLOAT, my_rank + 1,
                         tag_computing, MPI_COMM_WORLD);
            }
            if (my_rank > 0) {
      //          fprintf(stderr, "\tProcess %d -> Process %d\n", my_rank, my_rank - 1);
                MPI_Send(&my_ys_temp[first_index_to_compute], 1, MPI_FLOAT, my_rank - 1,
                         tag_computing, MPI_COMM_WORLD);
            }
            if (my_rank < processes - 1) {
                MPI_Recv(&my_ys_temp[my_points_num - 1], 1, MPI_FLOAT, my_rank + 1, 
                         tag_computing, MPI_COMM_WORLD, NULL);
            } 
            if (my_rank > 0) {
                MPI_Recv(&my_ys_temp[0], 1, MPI_FLOAT, my_rank - 1,
                         tag_computing, MPI_COMM_WORLD, NULL);
            }
        } else {
            if (my_rank > 0) {
                MPI_Recv(&my_ys_temp[0], 1, MPI_FLOAT, my_rank - 1,
                         tag_computing, MPI_COMM_WORLD, NULL);
            }
            if (my_rank < processes - 1) {
                MPI_Recv(&my_ys_temp[my_points_num - 1], 1, MPI_FLOAT, my_rank + 1, 
                         tag_computing, MPI_COMM_WORLD, NULL);
            }
            if (my_rank > 0) {
        //        fprintf(stderr, "\tProcess %d -> Process %d\n", my_rank, my_rank - 1);
                MPI_Send(&my_ys_temp[first_index_to_compute], 1, MPI_FLOAT, my_rank - 1,
                         tag_computing, MPI_COMM_WORLD);
            }
            if (my_rank < processes - 1) {
          //      fprintf(stderr, "\tProcess %d -> Process %d\n", my_rank, my_rank + 1);
                MPI_Send(&my_ys_temp[last_index_to_compute], 1, MPI_FLOAT, my_rank + 1,
                         tag_computing, MPI_COMM_WORLD);
            }            
        }

        // Swap the pointers
        float *t = my_ys;
        my_ys = my_ys_temp;
        my_ys_temp = t;
    }

    // 5. Collect the partial results 
    if (my_rank == 0) {
#ifdef DEBUG
        fprintf(stderr, "Started gathering results...\n");
#endif
        for (int i = 0; i <= my_points_end - 1; i++) {
            ys[i] = my_ys[i];
        }

        for (int rank = 1; rank < processes - 1; rank++) {
            MPI_Recv(ys + points_begin[rank] + 1, points_num[rank] - 2,
                     MPI_FLOAT, rank, tag_gathering, MPI_COMM_WORLD, NULL);
        }

        if (processes > 1)
            MPI_Recv(ys + points_begin[processes - 1] + 1, points_num[processes - 1] - 1,
                     MPI_FLOAT, processes - 1, tag_gathering, MPI_COMM_WORLD, NULL);
    } else {
        int count = my_points_num - 2;
        if (my_rank == processes - 1)
            count = my_points_num - 1;
        MPI_Send(&my_ys[1], count, MPI_FLOAT, 0, tag_gathering, MPI_COMM_WORLD);
    }

    // 6. Print the result out
    if (my_rank == 0) {
        for (int i = 0; i < points; i++) {
            float x = interval_begin + i * (interval_end - interval_begin) / points;
            float temperature = ys[i];
            fprintf(stdout, "%.8f %.8f \n", x, temperature);
        }

        free(ys);
    }

    fprintf(stderr, "Process: %d\n", my_rank);

    free(points_num);
    free(points_begin);
    free(points_end);

    free(buf1);
    free(buf2);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

