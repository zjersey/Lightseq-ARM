#include "profiler.h"
#include "omp.h"

void Profiler::set_start(const std::string& name) {
    double start_time = omp_get_wtime();
    record_start[name] = start_time;
}

double Profiler::set_end(const std::string& name) {
    double end_time = omp_get_wtime();
    double start_time = record_start[name];
    double duration = (end_time - start_time) * 1000;
    printf("duration, %s: %lf ms\n", name.c_str(), duration);
    return duration;
}