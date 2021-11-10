#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <numeric>
#include <future>

#include "struct_mapping.h"

struct Count {
    std::string name;
    std::vector<double> time_ms;
    std::vector<double> speed;
};

#ifndef __cplusplus
#ifndef _MSC_VER
#include <cstdalign>
#define _aligned_free free
#else
#define alignas(x) __declspec(align(x))
#define aligned_alloc(al, sz) _aligned_malloc((sz),(al))
#endif
#else
#ifndef _MSC_VER
#define _aligned_free free
#endif
#endif


#define STEPS 2000000000

double func(double x)
{
    return x * x;
}

typedef double (*f_t) (double);

typedef struct partial_sum_t_
{
    alignas(64) double result;
} partial_sum_t;

typedef struct experiment_result
{
    double result;
    double time_ms;
} experiment_result;

typedef double (*I_t)(double, double, f_t);

experiment_result run_experiment(I_t I)
{
    double t0 = omp_get_wtime();
    double R = I(-1, 1, func);
    double t1 = omp_get_wtime();
    return { R, t1 - t0 };
}

static unsigned num_threads = std::thread::hardware_concurrency();
unsigned get_num_threads()
{
    return num_threads;
}

void set_num_threads(unsigned t)
{
    num_threads = t;
    omp_set_num_threads(t);
}

void show_experiment_results_cli(I_t I, std::string name)
{
    double T1;
    std::cout << name << std::endl;
    printf("%10s\t%10s\t%10s\n", "Result", "Time_ms", "Speed");
    for (int T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        set_num_threads(T);
        R = run_experiment(I);
        if(T == 1)
            T1 = R.time_ms;
        printf("%10g\t%10g\t%10g\n", R.result, R.time_ms, T1 / R.time_ms);
    };
    std::cout << '\n';
};

void show_experiment_results_py(I_t I, std::string name)
{
    using namespace std;

    ofstream out;

    out.open("python/graphics/1.json");
    if (out.is_open())
    {
        struct_mapping::reg(&Count::name, "name");
        struct_mapping::reg(&Count::time_ms, "time_ms");
        struct_mapping::reg(&Count::speed, "speed");

        Count Count;


        Count.name = name;

        double T1;
        for (int T = 1; T <= omp_get_num_procs(); ++T) {
            experiment_result R;
            omp_set_num_threads(T);
            R = run_experiment(I);
            Count.time_ms.push_back(R.time_ms);
            if (T == 1)
                T1 = R.time_ms;
            double speed = T1 / R.time_ms;
            Count.speed.push_back(speed);
        }

        std::basic_ostringstream<char> json_data;
        struct_mapping::map_struct_to_json(Count, json_data, "  ");


        out << json_data.str() << std::endl;
    }

    std::cout << system("ls -la") << std::endl;
    system("/usr/local/bin/python3 python/main.py");
};

double integrate_crit(double a, double b, f_t f)
{
    double Result = 0;
    double dx = (b - a) / STEPS;
#pragma omp parallel shared(Result)
    {
        double R = 0;
        unsigned t = omp_get_thread_num();
        unsigned T = (unsigned)omp_get_num_threads();
        for (unsigned i = t; i < STEPS; i += T)
        {
            R += f(i * dx + a);
        }
#pragma omp critical
        Result += R;
    }
    return Result * dx;
}

double integrate_cpp_mtx(double a, double b, f_t f)
{
    using namespace std;
    unsigned T = get_num_threads();
    vector <thread> threads;
    mutex mtx;
    double Result = 0;
    double dx = (b - a) / STEPS;

    for (unsigned t = 0; t < T; ++t)
        threads.emplace_back([=, &Result, &mtx]()
                             {
            double R = 0;
            for (unsigned i = t; i < STEPS; i += T)
            {
                R += f(i * dx + a);
            }

            {
                scoped_lock lock{ mtx };
                Result += R;
            }
        });

    for (auto& thr : threads)
        thr.join();

    return Result * dx;
}

double integrate(double a, double b, f_t f)
{
    unsigned T;
    double Result = 0;
    double dx = (b - a) / STEPS;
    double* Accum;
#pragma omp parallel shared(Accum, T)
    {
        unsigned t = (unsigned int)omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned)omp_get_num_threads();
            Accum = (double*)calloc(T, sizeof(double));
        }

        for (unsigned i = t; i < STEPS; i += T)
            Accum[t] += f(dx * i + a);
    }

    for (unsigned int i = 0; i < T; i++)
        Result += Accum[i];

    return Result * dx;
}

double integrate_aligned(double a, double b, f_t f)
{
    unsigned T;
    double Result = 0;
    double dx = (b - a) / STEPS;
    partial_sum_t* Accum;
#pragma omp parallel shared(Accum, T)
    {
        unsigned t = (unsigned)omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned)omp_get_num_threads();
            Accum = (partial_sum_t*) aligned_alloc(alignof(partial_sum_t), T * sizeof(partial_sum_t_));
        }

        Accum[t].result = 0;
        for (unsigned i = t; i < STEPS; i += T)
            Accum[t].result += f(dx * i + a);
    }

    for (unsigned int i = 0; i < T; i++)
        Result += Accum[i].result;

    _aligned_free(Accum);

    return Result * dx;
}

double integrate_reduction(double a, double b, f_t f)
{
    double Result = 0.0;
    double dx = (b - a) / STEPS;
    int i;
#pragma omp parallel for reduction(+:Result)
    for (i = 0; i < STEPS; i++)
    {
        Result += f(dx * i + a);
    }

    return Result * dx;
}

double integrate_cpp(double a, double b, f_t f)
{
    double Result = 0;
    double dx = (b - a) / STEPS;
    using namespace std;
    unsigned T = get_num_threads();
    auto vec = vector(T, partial_sum_t{ 0.0 });
    vector <thread> threads;

    auto threads_proc = [=, &vec](auto t) {
        for (unsigned i = t; i < STEPS; i += T)
            vec[t].result += f(dx * i + a);
    };

    for (unsigned int t = 1; t < T; t++)
        threads.emplace_back(threads_proc,t);

    threads_proc(0);

    for (auto &thread:threads)
        thread.join();

    for (auto &elem:vec)
        Result += elem.result;

    return Result * dx;
}

double integrate_omp_for(double a, double b, f_t f)
{
    double result = 0;
    double dx = (b - a) / STEPS;

#pragma omp parallel
    {
        unsigned int t = (unsigned int)omp_get_thread_num();
        unsigned int T = (unsigned int)omp_get_num_threads();
        double accum = 0;
        for (int i = t; i < STEPS; i += T)
        {
            double val = f(dx * i + a);
            accum  += val;
        }
#pragma omp atomic
        result += accum;
    }

    return result * dx;
}

double integrate_cpp_reduction(double a, double b, f_t f)
{
    using namespace std;
    unsigned T = get_num_threads();
    double dx = (b - a) / STEPS;
    vector <thread> threads;
    atomic <double> Result{ 0.0 };
    auto threads_proc = [dx, &Result, f, a, T](auto t)
    {

        double R = 0;
        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }

        Result = Result + R;
    };

    for (unsigned int t = 1; t < T; t++)
        threads.emplace_back(threads_proc, t);

    threads_proc(0);

    for (auto& thread : threads)
        thread.join();

    return Result * dx;
}

typedef unsigned (*FibonacciFunction)(unsigned);

typedef struct fibonacci_result
{
    double result;
    double time_ms;
} fibonacci_result;

experiment_result run_experiment(FibonacciFunction f)
{
    double t0 = omp_get_wtime();
    double R = f(10);
    double t1 = omp_get_wtime();
    return { R, t1 - t0 };
}

void show_fibonacci_experiment_results_cli(FibonacciFunction f, std::string name)
{
    double t1;
    std::cout << name << std::endl;
    printf("%10s\t%10s\t%10s\n", "Result", "Time_ms", "Speed");
    for (int t = 1; t <= omp_get_num_procs(); ++t) {
        experiment_result result;
        set_num_threads(t);
        result = run_experiment(f);
        if (t == 1)
            t1 = result.time_ms;
        printf("%10g\t%10g\t%10g\n", result.result, result.time_ms, t1 / result.time_ms);
    };
    std::cout << std::endl;
}

unsigned fibonacci(unsigned n) {
    if (n < 2)
        return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

unsigned fibonacci_omp(unsigned n) {
    if (n < 2)
        return n;
    unsigned x1, x2;
#pragma omp task
    {
        x1 = fibonacci_omp(n - 1);
    }
#pragma omp task
    {
        x2 = fibonacci_omp(n - 2);
    }
#pragma omp taskwait
    return x1 + x2;
}

unsigned fibonacci_cpp(unsigned n) {
    if (n < 2)
        return n;

    auto x1 = std::async(fibonacci_cpp, n - 1);
    auto x2 = std::async(fibonacci_cpp, n - 2);

    return x1.get() + x2.get();
}

int main()
{

    std::cout << alignof(partial_sum_t) << std::endl;
    std::cout << get_num_threads() << std::endl;

    set_num_threads(16);
//    fibonacci_omp(256);
    show_fibonacci_experiment_results_cli(fibonacci, "fibonacci single thread");
    show_fibonacci_experiment_results_cli(fibonacci_omp, "fibonacci omp");
    show_fibonacci_experiment_results_cli(fibonacci_cpp, "fibonacci cpp");

    //show_experiment_results_py(integrate_cpp, "integrate_cpp");

    return 0;
}
