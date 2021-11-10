#include <barrier>
#include <cstdlib>
#include <thread>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <iostream>

using namespace std;

#if defined(__GNUC__) && __GNUC__ <= 10
namespace std {
constexpr std::size_t hardware_constructive_interference_size = 64u;
constexpr std::size_t hardware_destructive_interference_size = 64u;
}
#endif

std::size_t ceil_div(std::size_t x, std::size_t y)
{
    return (x + y - 1) / y;
}

// TODO:
static unsigned num_treads = std::thread::hardware_concurrency();
unsigned get_num_threads()
{
    return num_treads;
}

void set_num_threads(unsigned t)
{
    num_treads = t;
    omp_set_num_threads(t);
}

// MARK: - Reduce

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, std::size_t n, BinaryFn f, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(std::hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =

    std::vector<reduction_partial_result_t>(std::thread::hardware_concurrency(),
                                            reduction_partial_result_t{zero});
    constexpr std::size_t k = 2;
    std::barrier<> bar {T};

    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        std::size_t Mt = K / T;
        std::size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        std::size_t mt = Mt * k;
        std::size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(std::size_t i = it1; i < it2; i++)
            accum = f(accum, V[i]);

        reduction_partial_results[t].value = accum;

#if 0
        std::size_t s = 1;
        while(s < T)
        {
            bar.arrive_and_wait();
            if((t % (s * k)) && (t + s < T))
            {
                reduction_partial_results[t].value = f(reduction_partial_results[t].value,
                                                       reduction_partial_results[t + s].value);
                s *= k;
            }
        }
#else
        for(std::size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next) //TODO assume k = 2
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = f(reduction_partial_results[t].value,
                                                       reduction_partial_results[t + s].value);
        }
#endif
    };


    std::vector<std::thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();

    return reduction_partial_results[0].value;
}

#include <type_traits>

template <class ElementType, class UnaryFn, class BinaryFn>
#if 0
requires {
    std::is_invocable_r_v<UnaryFn, ElementType, ElementType> &&
    std::is_invocable_r_v<BinaryFn, ElementType, ElementType, ElementType>
}
#endif
ElementType reduce_range(ElementType a, ElementType b, std::size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(std::hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
    std::vector<reduction_partial_result_t>(std::thread::hardware_concurrency(), reduction_partial_result_t{zero});

    std::barrier<> bar{T};
    constexpr std::size_t k = 2;
    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        double dx = (b - a) / n;
        std::size_t Mt = K / T;
        std::size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        std::size_t mt = Mt * k;
        std::size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(std::size_t i = it1; i < it2; i++)
            accum = reduce_2(accum, get(a + i*dx));

        reduction_partial_results[t].value = accum;

        for(std::size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next) //TODO assume k = 2
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = reduce_2(reduction_partial_results[t].value,
                                                              reduction_partial_results[t + s].value);
        }
    };

    std::vector<std::thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();
    return reduction_partial_results[0].value;
}

// MARK: - Randomize

typedef double (*RandomizerFunction)(unsigned, unsigned *, size_t, unsigned, unsigned);

typedef struct experiment_result
{
    double result;
    double time_ms;
} experiment_result;

experiment_result run_experiment(RandomizerFunction f) {
    double t0 = omp_get_wtime();

    size_t arrayLength = 100000000;
    unsigned *array = (unsigned *)malloc(arrayLength * sizeof(unsigned));

    double result = f(31231231, array, arrayLength, 100, 200);
    double t1 = omp_get_wtime();
    return { result, t1 - t0 };
}

void show_experiment_results_cli(RandomizerFunction f, std::string name) {
    double t1;
    std::cout << name << std::endl;
    printf("%10s\t%10s\t%10s\n", "Result", "Time_ms", "Speed");
    for (int t = 1; t <= omp_get_num_procs(); ++t) {
        experiment_result result;
        set_num_threads(t);
        result = run_experiment(f);
        if (t == 1) {
            t1 = result.time_ms;
        }
        printf("%10g\t%10g\t%10g\n", result.result, result.time_ms, t1 / result.time_ms);
    }
    std::cout << std::endl;
}

double randomize_array_single(unsigned seed, unsigned *v, size_t n, unsigned min, unsigned max) {
    uint64_t a = 6364136223846793005;
    unsigned b = 1;

    uint64_t prev = seed;
    uint64_t sum = 0;
    for (unsigned i = 0; i < n; i++) {
        uint64_t nextValue = a * prev + b;
        v[i] = (nextValue % (max - min + 1)) + min;
        prev = nextValue;
        sum += v[i];
    }

    return (double)sum/(double)n;
}

uint64_t pow(uint64_t num, unsigned i, uint64_t c) {
    uint64_t result = 1;

    for (int j = 0; j < i; j++) {
        result *= num;
    }

    return result;
}

uint64_t sum(unsigned start, unsigned end, uint64_t num, uint64_t c) {
    uint64_t sum = 0;
    for (unsigned i = 0; i <= end; i++) {
        sum += pow(num, i, c);
    }
    return sum;
}

double randomize_array_omp(unsigned seed, unsigned *v, size_t n, unsigned min, unsigned max) {
    uint64_t a = 6364136223846793005;
    unsigned b = 1;

    uint64_t result = 0;

    unsigned T = (unsigned)get_num_threads();

    uint64_t A = pow(a, T, max - min + 1);
    uint64_t B = b * sum(0, T - 1, a, max - min + 1);

    uint64_t prev = seed;

    uint64_t *vp = new uint64_t[T];

    for (unsigned i = 0; i < T; i++) {
        uint64_t next = a * prev + b;
        v[i] = (next % (max - min + 1)) + min;
        vp[i] = next;
        prev = next;
    }

#pragma omp parallel shared(v)
    {
        unsigned t = (unsigned int)omp_get_thread_num();
        uint64_t prev = vp[t];
        for (unsigned i = t + T; i < n; i += T) {
            uint64_t next = A * prev + B;
            prev = next;
            v[i] = (next % (max - min + 1)) + min;
        }
    }

    return (double)reduce_vector(v, n, [](double x, double y) {return x + y;}, 0u) / (double)n;
}

#define MIN 100
#define MAX 200
#define SEED 1312312

uint64_t* getLUTA(unsigned size, uint64_t a){
    uint64_t res[size+1];
    res[0] = 1;
    for (unsigned i=1; i<=size; i++) res[i] = res[i-1] * a;
    return res;
}

uint64_t* getLUTB(unsigned size, uint64_t* a, uint64_t b){
    uint64_t res[size];
    res[0] = b;
    for (unsigned i=1; i<size; i++){
        uint64_t acc = 0;
        for (unsigned j=0; j<=i; j++){
            acc += a[j];
        }
        res[i] = acc*b;
    }
    return res;
}

uint64_t getA(unsigned size, uint64_t a){
    uint64_t res = 1;
    for (unsigned i=1; i<=size; i++) res = res * a;
    return res;
}

uint64_t getB(unsigned size, uint64_t a){
    uint64_t* acc = new uint64_t(size);
    uint64_t res = 1;
    acc[0] = 1;
    for (unsigned i=1; i<=size; i++){
        for (unsigned j=0; j<i; j++){
            acc[i] = acc[j] * a;
        }
        res += acc[i];
    }
    free(acc);
    return res;
}

double randomize_arr(unsigned* V, size_t n){
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    unsigned T;
    //    uint64_t* LUTA;
    //    uint64_t* LUTB;
    uint64_t LUTA;
    uint64_t LUTB;
    uint64_t sum = 0;

#pragma omp parallel shared(V, T, LUTA, LUTB)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) get_num_threads();
            //            LUTA = getLUTA(n, a);
            //            LUTB = getLUTB(n, LUTA, b);
            LUTA = getA(T, a);
            LUTB = getB((T - 1), a)*b;
        }
        uint64_t prev = SEED;
        uint64_t cur;

        for (unsigned i=t; i<n; i += T){
            if (i == t){
                cur = getA(i+1, a)*prev + getB(i, a) * b;
            } else {
                cur = LUTA*prev + LUTB;
            }
            //            cur = LUTA[i+1]*prev + LUTB[i];
            V[i] = (cur % (MAX - MIN + 1)) + MIN;
            prev = cur;
        }
    }

    for (unsigned i=0; i<n;i++)
        sum += V[i];

    return (double)sum/(double)n;
}

#pragma mark - Main

int main()
{
    unsigned V[13];

    // set_num_threads(1);
    size_t arrayLength = 1000000000000;
    unsigned *singleThreadedArray = (unsigned *)malloc(sizeof(unsigned) * arrayLength);
    unsigned *multiThreadedArray = (unsigned *)malloc(sizeof(unsigned) * arrayLength);
    unsigned *thirdArray = (unsigned *)malloc(sizeof(unsigned) * arrayLength);

//    show_experiment_results_cli(randomize_array_single, "randomize single");

    set_num_threads(16);
    show_experiment_results_cli(randomize_array_omp, "randomize omp");

    for(unsigned i = 0; i < std::size(V); i++)
        V[i] = i + 1;
    std::cout << "Average: " << reduce_vector(V, std::size(V), [](auto x, auto y) {return x + y;}, 0u)/ (double)std::size(V) << '\n';


}
