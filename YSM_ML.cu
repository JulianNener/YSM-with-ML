#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>
#include <sstream>
#include <map>
#include <cub/cub.cuh>

#include <time.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLOBAL_SEED 1234567890

#define N 100000
#define N_TO_TRAIN 10000
#define W_TOT 1
#define W_MIN 3e-17

#define N_INPUTS 1
#define N_LAYERS 1
#define N_OUTPUTS 1


__constant__ unsigned d_layer_sizes[N_LAYERS];
__constant__ unsigned d_n_nodes_ptr[1];
__constant__ unsigned d_n_wgts_ptr[1];
unsigned h_layer_sizes[N_LAYERS];
unsigned h_n_nodes_ptr[1];
unsigned h_n_wgts_ptr[1];


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void printMemInfo() {
  size_t freemem;
  size_t totalmem;
  cudaMemGetInfo(&freemem, &totalmem);
  printf("\r%4.2fMB/ %4.2fMB (Free/ Total)       ",
    freemem / (1024 * 1024.0),
    totalmem / (1024 * 1024.0));
}


using namespace std;
using namespace thrust::placeholders;

struct agents
{
	float* w;
	float* r;
};

struct params
{
	float* a;
	float* b;
	float* w;
};

struct params_new
{
	float* b;
	float* w;
};

typedef float(*fun_ptr_t)(float);
typedef float(*mutagene_fun_ptr_t)(float, float);


//Index = xn ( D1 * ... * D{n-1} ) + x{n-1} ( D1 * ... * D{n-2} ) + ... + x2 * D1 + x1
__host__ __device__ int idx_w(int agent_idx, int layer_idx, int neuron_idx, int weight_idx) 
{
	#ifdef  __CUDA_ARCH__
		unsigned* layer_sizes = d_layer_sizes;
		unsigned n_wgts = *d_n_wgts_ptr;
	#else
		unsigned* layer_sizes = h_layer_sizes;
		unsigned n_wgts = *h_n_wgts_ptr;
	#endif
	
	unsigned layer_jump = 0;
	unsigned neuron_jump = N_INPUTS*neuron_idx;
	if(layer_idx > 0)
	{
		layer_jump = N_INPUTS*layer_sizes[0];
		for(int i = 1; i < layer_idx; ++i)
			layer_jump += layer_sizes[i-1]*layer_sizes[i];

		neuron_jump = layer_sizes[layer_idx-1]*neuron_idx;
	}

	return agent_idx*(n_wgts) + layer_jump + neuron_jump + weight_idx;
}

__host__ __device__ int idx_b(int agent_idx, int layer_idx, int neuron_idx) {
	#ifdef  __CUDA_ARCH__
		unsigned* layer_sizes = d_layer_sizes;
		unsigned n_nodes = *d_n_nodes_ptr;
	#else
		unsigned* layer_sizes = h_layer_sizes;
		unsigned n_nodes = *h_n_nodes_ptr;
	#endif
	
	unsigned layer_jump = 0;
	for(int i = 0; i < layer_idx; ++i) layer_jump += layer_sizes[i];

	return agent_idx*(n_nodes - N_INPUTS) + layer_jump + neuron_idx;
}

__host__ __device__ int idx_a(int agent_idx, int layer_idx, int neuron_idx) {
	#ifdef  __CUDA_ARCH__
		unsigned* layer_sizes = d_layer_sizes;
		unsigned n_nodes = *d_n_nodes_ptr;
	#else
		unsigned* layer_sizes = h_layer_sizes;
		unsigned n_nodes = *h_n_nodes_ptr;
	#endif

	unsigned layer_jump = 0;
	if(layer_idx > 0) 
	{
		layer_jump = N_INPUTS;
		for(int i = 0; i < layer_idx-1; ++i) layer_jump += layer_sizes[i];
	}
	
	return agent_idx*(n_nodes) + layer_jump + neuron_idx;
}

__device__ void calculate_activations(fun_ptr_t fun, params* par, int agent_idx)
{
    float sum, b;
	int n_neurons, n_neurons_prev;

	n_neurons = d_layer_sizes[0];
    for(int i = 0; i < n_neurons; ++i)
    {
        sum = 0;
        b = par->b[idx_b(agent_idx, 0, i)];
        for(int j = 0; j < N_INPUTS; ++j)
			sum += par->w[idx_w(agent_idx, 0, i, j)] * par->a[idx_a(agent_idx, 0, j)];		
		
        par->a[idx_a(agent_idx, 1, i)] = fun(sum + b);
    }

    for(int i = 1; i < N_LAYERS; ++i)
    {
        n_neurons_prev = d_layer_sizes[i-1];
        n_neurons = d_layer_sizes[i];
        for(int j = 0; j < n_neurons; ++j)
        {
            sum = 0;
            b = par->b[idx_b(agent_idx, i, j)];
            for(int k = 0; k < n_neurons_prev; ++k)
				sum += par->w[idx_w(agent_idx, i, j, k)] * par->a[idx_a(agent_idx, i, k)];

            par->a[idx_a(agent_idx, i+1, j)] = fun(sum + b);			
        }
    }

	n_neurons_prev = d_layer_sizes[N_LAYERS-1];
    for(int i = 0; i < N_OUTPUTS; ++i)
    {
        sum = 0;
        b = par->b[idx_b(agent_idx, N_LAYERS, i)];
        for(int j = 0; j < n_neurons_prev; ++j)
			sum += par->w[idx_w(agent_idx, N_LAYERS, i, j)] * par->a[idx_a(agent_idx, N_LAYERS, j)];

        par->a[idx_a(agent_idx, N_LAYERS+1, i)] =  fun(sum + b);
    }
}

__device__ inline void set_inputs(params* par, int agent_idx, float* input_list) 
{
    for(int i = 0; i < N_INPUTS; ++i)
        par->a[idx_a(agent_idx, 0, i)] = input_list[i];
}

__device__ void mutate(int agent_idx, params_new* par_new, float mutagene, curandState_t* states, int id)
{
	int n_neurons, n_neurons_prev;

	n_neurons = d_layer_sizes[0];
    for(int i = 0; i < n_neurons; ++i)
    {
        for(int j = 0; j < N_INPUTS; ++j)
            par_new->w[idx_w(agent_idx, 0, i, j)] += (2*curand_uniform(&states[id]) - 1) * mutagene;
			
        par_new->b[idx_b(agent_idx, 0, i)] += (2*curand_uniform(&states[id]) - 1) * mutagene;
    }
    
    for(int i = 1; i < N_LAYERS; ++i)
    {
        n_neurons_prev = d_layer_sizes[i-1];
        n_neurons = d_layer_sizes[i];
        for(int j = 0; j < n_neurons; ++j)
        {
            for(int k = 0; k < n_neurons_prev; ++k)
                par_new->w[idx_w(agent_idx, i, j, k)] += (2*curand_uniform(&states[id]) - 1) * mutagene;

           par_new->b[idx_b(agent_idx, i, j)] += (2*curand_uniform(&states[id]) - 1) * mutagene;
        }
    }

	n_neurons_prev = d_layer_sizes[N_LAYERS-1];
    for(int i = 0; i < N_OUTPUTS; ++i)
    {
        for(int j = 0; j < n_neurons_prev; ++j)
            par_new->w[idx_w(agent_idx, N_LAYERS, i, j)] += (2*curand_uniform(&states[id]) - 1) * mutagene;

        par_new->b[idx_b(agent_idx, N_LAYERS, i)] += (2*curand_uniform(&states[id]) - 1) * mutagene;
    }
}

__global__ void init_rands(unsigned seed, curandState_t* states) 
{

	int id = threadIdx.x + (blockIdx.x * blockDim.x);

  /* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
				 id, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				 &states[id]);
}


__device__ void fair_rule(float& w1, float& w2, float r1, float r2, curandState_t* states, int id)
{
	float dw = (r1*w1 < r2*w2) ? r1*w1 : r2*w2;
	float p = curand_uniform(&states[id]);

	if(p < 0.5)
	{
		w1 += dw;
		w2 -= dw;
	}
	else
	{
		w1 -= dw;
		w2 += dw;
	}

	if(w1 < W_MIN) w1 = 0;
	if(w2 < W_MIN) w2 = 0;
}

__device__ inline bool is_trained(int ag_idx) {
	return (ag_idx < N_TO_TRAIN) ? true : false;
}

__device__ inline void update_risk(agents* ag, params* par, int agent_idx) {
    ag->r[agent_idx] = par->a[idx_a(agent_idx, N_LAYERS+1, 0)];
}

__global__ void trade_once(agents* ag, int* idx, params* par, fun_ptr_t fun, float* p_rep, curandState_t* states)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x); 
	if(id < N/2)
	{
		int idx1 = idx[2*id], idx2 = idx[2*id+1];

		float w1 = ag->w[idx1], w2 = ag->w[idx2];
		if( (w1 > 0) && (w2 > 0) )
		{
			float r1 = ag->r[idx1], r2 = ag->r[idx2];
			if(is_trained(idx1))
			{
				float input_list1[] = { -2*logf(w1/W_MIN)/logf(W_MIN) - 1 };
				// float input_list1[] = { r1 };
				set_inputs(par, idx1, input_list1);
				calculate_activations(fun, par, idx1);
				update_risk(ag, par, idx1);
			}

			if(is_trained(idx2))
			{
				float input_list2[] = { -2*logf(w2/W_MIN)/logf(W_MIN) - 1 };
				// float input_list2[] = { r2 };
				set_inputs(par, idx2, input_list2);
				calculate_activations(fun, par, idx2);
				update_risk(ag, par, idx2);
			}

			r1 = ag->r[idx1], r2 = ag->r[idx2];
			fair_rule(w1, w2, r1, r2, states, id);
						
			ag->w[idx1] = w1;
			ag->w[idx2] = w2;
		}

		if(is_trained(idx1) && w1 > 0) p_rep[idx1] += w1;
		if(is_trained(idx2) && w2 > 0) p_rep[idx2] += w2;
	}
}

__global__ void distribute_taxes(float dw, int max_idx, int* ag_idx_wSorted, float* w)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x); 
	if(id < max_idx)
		w[ag_idx_wSorted[id]] += dw;
}

__global__ void tax_the_ppl(float* w, float lambda)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x); 
	if(id < N)
		w[id] -= lambda*w[id];
}


struct rep_fun_line
{
	float x_min, x_max;
	
	rep_fun_line(float in_min, float in_max) 
	{
		x_min = in_min;
		x_max = in_max;
	}

	__host__ __device__ 
	float operator()(float x) {
		return (x-x_min)/(x_max-x_min);
	}
};

void set_reproduction_probs(float* p_rep)
{
	thrust::pair<float*, float*> p_rep_extrema = thrust::minmax_element(thrust::device, p_rep, p_rep + N_TO_TRAIN);

	thrust::device_vector<float> d_minmax_elem(2);
	d_minmax_elem[0] = *p_rep_extrema.first;
	d_minmax_elem[1] = *p_rep_extrema.second;

	thrust::host_vector<float> h_minmax_elem = d_minmax_elem;

	thrust::transform(thrust::device, p_rep, p_rep + N_TO_TRAIN, p_rep, rep_fun_line(h_minmax_elem[0], h_minmax_elem[1]));
}

__device__ void copyNN_to_new(int ag_idx, params* par, int ag_idx_new, params_new* par_new)
{
	int n_neurons, n_neurons_prev;

	n_neurons = d_layer_sizes[0];
    for(int i = 0; i < n_neurons; ++i)
    {
        for(int j = 0; j < N_INPUTS; ++j)
			par_new->w[idx_w(ag_idx_new, 0, i, j)] = par->w[idx_w(ag_idx, 0, i, j)];

		par_new->b[idx_b(ag_idx_new, 0, i)] = par->b[idx_b(ag_idx, 0, i)];
    }

    for(int i = 1; i < N_LAYERS; ++i)
    {
        n_neurons_prev = d_layer_sizes[i-1];
        n_neurons = d_layer_sizes[i];
        for(int j = 0; j < n_neurons; ++j)
        {
            for(int k = 0; k < n_neurons_prev; ++k)
				par_new->w[idx_w(ag_idx_new, i, j, k)] = par->w[idx_w(ag_idx, i, j, k)];

            par_new->b[idx_b(ag_idx_new, i, j)] = par->b[idx_b(ag_idx, i, j)];
        }
    }
	
	n_neurons_prev = d_layer_sizes[N_LAYERS-1];
    for(int i = 0; i < N_OUTPUTS; ++i)
    {
        for(int j = 0; j < n_neurons_prev; ++j)
			par_new->w[idx_w(ag_idx_new, N_LAYERS, i, j)] = par->w[idx_w(ag_idx, N_LAYERS, i, j)];

		par_new->b[idx_b(ag_idx_new, N_LAYERS, i)] = par->b[idx_b(ag_idx, N_LAYERS, i)];
    }
}

__global__ void kernel_reproduce(int* idx_new_last, float* p_rep, params* par, params_new* par_new, float mutagene, curandState_t* states)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	if(id < N_TO_TRAIN)
	{
		if(curand_uniform(&states[id]) < p_rep[id])
		{
			int idx_new = atomicAdd(idx_new_last, 1); //atomicAdd devuelve el valor previo
			if(idx_new < N_TO_TRAIN)
			{
				copyNN_to_new(id, par, idx_new, par_new);
				mutate(idx_new, par_new, mutagene, states, id);
			}
		}
	}
}

int cnt_brains(string brains_filename)
{
    string line;
    ifstream in{brains_filename};
    int N_ag = 0;

    while(getline(in, line))
        if(line.empty())
            ++N_ag;

    return N_ag;
}

void load_brain_neurons(unsigned* layer_sizes, thrust::host_vector<float>& biases, thrust::host_vector<float>& weights, string filename)
{
	ifstream in{filename};
	assert(in);
	int N_brains = cnt_brains(filename);
	assert(N_brains == N_TO_TRAIN);
	
	for(int j = 0; j < N_TO_TRAIN; ++j)
	{
		int n_neurons = layer_sizes[0];
		for(int k = 0; k < n_neurons; ++k)
		{
			for(int l = 0; l < N_INPUTS; ++l)
				in >> weights[idx_w(j, 0, k, l)];
				
			in >> biases[idx_b(j, 0, k)];
		}

		int n_neurons_prev;
		for(int i = 1; i < N_LAYERS; ++i)
		{
			n_neurons_prev = layer_sizes[i-1];
			n_neurons = layer_sizes[i];
			for(int k = 0; k < n_neurons; ++k)
			{
				for(int l = 0; l < n_neurons_prev; ++l)
					in >> weights[idx_w(j, i, k, l)];

				in >> biases[idx_b(j, i, k)];
			}
		}

		n_neurons_prev = layer_sizes[N_LAYERS-1];
		for(int k = 0; k < N_OUTPUTS; ++k)
		{
			for(int l = 0; l < n_neurons_prev; ++l)
				in >> weights[idx_w(j, N_LAYERS, k, l)];

			in >> biases[idx_b(j, N_LAYERS, k)];
		}
	}
}

void print_brain_neurons(unsigned* layer_sizes, thrust::host_vector<float> biases, thrust::host_vector<float> weights, string filename)
{
	ofstream out{filename};
	
	for(int j = 0; j < N_TO_TRAIN; ++j)
	{
		int n_neurons = layer_sizes[0];
		for(int k = 0; k < n_neurons; ++k)
		{
			for(int l = 0; l < N_INPUTS; ++l)
				out << weights[idx_w(j, 0, k, l)] << " ";
				
			out << biases[idx_b(j, 0, k)] << endl;
		}

		int n_neurons_prev;
		for(int i = 1; i < N_LAYERS; ++i)
		{
			n_neurons_prev = layer_sizes[i-1];
			n_neurons = layer_sizes[i];
			for(int k = 0; k < n_neurons; ++k)
			{
				for(int l = 0; l < n_neurons_prev; ++l)
					out << weights[idx_w(j, i, k, l)] << " ";

				out << biases[idx_b(j, i, k)] << endl;
			}
		}

		n_neurons_prev = layer_sizes[N_LAYERS-1];
		for(int k = 0; k < N_OUTPUTS; ++k)
		{
			for(int l = 0; l < n_neurons_prev; ++l)
				out << weights[idx_w(j, N_LAYERS, k, l)] << " ";

			out << biases[idx_b(j, N_LAYERS, k)] << endl;
		}

		out << endl;
	}
	
}

typedef thrust::counting_iterator<int> CountingIterator;
typedef thrust::tuple<CountingIterator, float> tup;

struct mult_by_idx
{
	__device__
	float operator()(tup t)
	{
		int idx = *thrust::get<0>(t);
		return idx*thrust::get<1>(t);
	}
};


float get_gini(float* d_wealth)
{
	float* d_wealth_sorted;
	cudaMalloc(&d_wealth_sorted, sizeof(float)*N);

	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_wealth, d_wealth_sorted, N);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_wealth, d_wealth_sorted, N);
	
	CountingIterator count_begin(1);
	CountingIterator count_end(N+1);
	float sum = thrust::transform_reduce(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(count_begin, d_wealth_sorted)), 
														 thrust::make_zip_iterator(thrust::make_tuple(count_end, d_wealth_sorted + N)),
														 mult_by_idx(), 0.0f, thrust::plus<float>() );
	
	cudaFree(d_temp_storage);
	cudaFree(d_wealth_sorted);

	return 1 - 2.0/(N-1)*(N - sum);
}

void print_gini(float* wealth, int step, string filename)
{
	static ofstream out{filename};
	out << step << " " << get_gini(wealth) << endl;
}

string get_full_filename(unsigned* layer_sizes, int generations, float lambda, float p, float mutagene)
{
	string full_filename;
	for(int i = 0; i < N_LAYERS; ++i)
		full_filename += to_string(layer_sizes[i]) + "n";
	full_filename += "_";

	full_filename += to_string(generations) + "g_" + to_string(N_TO_TRAIN) + "t-" + to_string(N-N_TO_TRAIN) + "rc_" + "lamb" + to_string(lambda);
	full_filename.erase(full_filename.find_last_not_of('0') + 1, string::npos);
	full_filename += "p" + to_string(p);
	full_filename.erase(full_filename.find_last_not_of('0') + 1, string::npos);

	full_filename +="_mg" + to_string(mutagene);
	full_filename.erase(full_filename.find_last_not_of('0') + 1, string::npos);

	full_filename += ".txt";

	return full_filename;
}

int sample_agents_brains(unsigned* layer_sizes, int MCS_per_generation, int generations, float lambda, float p, 
						 float mutagene, fun_ptr_t activ_fun, unsigned* brains_generations_to_print,
						 bool load_brains, string filename_brains_in, 
						 bool calc_gini, int gini_generations, string filename_gini)
{
	dim3 nThreads(32);
	dim3 nBlocks_all( (N + nThreads.x - 1) / nThreads.x ); 
	dim3 nBlocks_half( (N/2 + nThreads.x - 1) / nThreads.x ); 
	dim3 nBlocks_reproduce( (N_TO_TRAIN + nThreads.x - 1) / nThreads.x );
	dim3 nBlocks_tax( (p*N + nThreads.x - 1) / nThreads.x );  

	/* alloc device constants */
	for(int i = 0; i < N_LAYERS; ++i)
		h_layer_sizes[i] = layer_sizes[i];
	cudaMemcpyToSymbol(d_layer_sizes, h_layer_sizes, sizeof(unsigned)*N_LAYERS);

	unsigned n_nodes = N_INPUTS + N_OUTPUTS;
	for(int i = 0; i < N_LAYERS; ++i)
		n_nodes += layer_sizes[i];
	*h_n_nodes_ptr = n_nodes;
	cudaMemcpyToSymbol(d_n_nodes_ptr, h_n_nodes_ptr, sizeof(unsigned));

	unsigned n_wgts = N_INPUTS*layer_sizes[0] + N_OUTPUTS*layer_sizes[N_LAYERS-1];
	for(int i = 1; i < N_LAYERS; ++i)
		n_wgts += layer_sizes[i-1]*layer_sizes[i];
	*h_n_wgts_ptr = n_wgts;
	cudaMemcpyToSymbol(d_n_wgts_ptr, h_n_wgts_ptr, sizeof(unsigned));
	/**************************/

	/* init random stuff */
	curandState_t* states;
	cudaMalloc((void**) &states, sizeof(curandState_t)*N);
	init_rands<<<nBlocks_all, nThreads>>>(GLOBAL_SEED, states);

	curandGenerator_t rng;
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(rng, 1234ULL);
	/*********************/

	/* alloc agents SoA */
	thrust::device_vector<float> d_risk(N), d_wealth(N);
	float* raw_wealth = thrust::raw_pointer_cast(&d_wealth[0]);
	float* raw_risk = thrust::raw_pointer_cast(&d_risk[0]);

	agents *d_agents;
	cudaMalloc(&d_agents, sizeof(*d_agents));
	cudaMemcpy(&(d_agents->r), &raw_risk, sizeof(d_agents->r), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_agents->w), &raw_wealth, sizeof(d_agents->w), cudaMemcpyHostToDevice);

	thrust::device_vector<float> d_p_rep(N_TO_TRAIN);
	float* raw_p_rep = thrust::raw_pointer_cast(&d_p_rep[0]);

	thrust::device_vector<int> d_idx_new_last(1);
	int* raw_idx_new_last = thrust::raw_pointer_cast(&d_idx_new_last[0]);
	/*******************/

	/* indices */
	thrust::device_vector<int> d_indices(N), d_indices_shuffled(N);
	thrust::sequence(d_indices.begin(), d_indices.end(), 0);
	int* raw_indices = thrust::raw_pointer_cast(&d_indices[0]);
	int* raw_indices_shuffled = thrust::raw_pointer_cast(&d_indices_shuffled[0]);

	thrust::device_vector<int> d_indices_sorted(N);
	int* raw_indices_sorted = thrust::raw_pointer_cast(&d_indices_sorted[0]);
	
	float *d_randoms, *d_randoms_sorted;
	cudaMalloc((void**) &d_randoms, sizeof(float)*N);
	cudaMalloc((void**) &d_randoms_sorted, sizeof(float)*N);
	/***********/

	/* alloc NN params SoA */
	unsigned biases_info_sz = (n_nodes-N_INPUTS) * N_TO_TRAIN; //biases
	unsigned weights_info_sz = n_wgts * N_TO_TRAIN; //weights
	unsigned activs_info_sz = n_nodes * N_TO_TRAIN; //activations

	thrust::host_vector<float> h_biases(biases_info_sz), h_weights(weights_info_sz);

	thrust::device_vector<float> d_activs(activs_info_sz), d_biases(biases_info_sz), d_weights(weights_info_sz);
	float* raw_activs = thrust::raw_pointer_cast(&d_activs[0]);
	float* raw_biases = thrust::raw_pointer_cast(&d_biases[0]);
	float* raw_weights = thrust::raw_pointer_cast(&d_weights[0]);

	thrust::device_vector<float> d_biases_new(biases_info_sz), d_weights_new(weights_info_sz);
	float* raw_biases_new = thrust::raw_pointer_cast(&d_biases_new[0]);
	float* raw_weights_new = thrust::raw_pointer_cast(&d_weights_new[0]);

	params *d_params;
	cudaMalloc((void **) &d_params, sizeof(*d_params));
	cudaMemcpy(&(d_params->a), &raw_activs, sizeof(d_params->a), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_params->b), &raw_biases, sizeof(d_params->b), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_params->w), &raw_weights, sizeof(d_params->w), cudaMemcpyHostToDevice);

	params_new *d_params_new;
	cudaMalloc((void **) &d_params_new, sizeof(*d_params_new));
	cudaMemcpy(&(d_params_new->b), &raw_biases_new, sizeof(d_params_new->b), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_params_new->w), &raw_weights_new, sizeof(d_params_new->w), cudaMemcpyHostToDevice);
	/***********************/

	/* init agents */
	curandGenerateUniform(rng, raw_risk, N);
	curandGenerateUniform(rng, raw_wealth, N);
	float sum_w_syst = thrust::reduce(d_wealth.begin(), d_wealth.end());
	thrust::transform(d_wealth.begin(), d_wealth.end(), d_wealth.begin(), _1/sum_w_syst);

	if(load_brains)
	{
		load_brain_neurons(layer_sizes, h_biases, h_weights, filename_brains_in);
		d_biases = h_biases;
		d_weights = h_weights;
	}
	else
	{
		curandGenerateUniform(rng, raw_biases, biases_info_sz);
		curandGenerateUniform(rng, raw_weights, weights_info_sz);
		thrust::transform(d_biases.begin(), d_biases.end(), d_biases.begin(), 2.*_1 - 1.);
		thrust::transform(d_weights.begin(), d_weights.end(), d_weights.begin(), 2.*_1 - 1.);
	}
	/***************/

	/* init radix sort de CUB (x2) */
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_randoms, d_randoms_sorted, raw_indices, raw_indices_shuffled, N, 0, sizeof(float)*8);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	void *d_temp_storage_2 = NULL;
	size_t temp_storage_bytes_2 = 0;
	thrust::device_vector<float> d_wealth_sorted(N);
	float* raw_wealth_sorted = thrust::raw_pointer_cast(&d_wealth_sorted[0]);

	cub::DeviceRadixSort::SortPairs(d_temp_storage_2, temp_storage_bytes_2, raw_wealth, raw_wealth_sorted, raw_indices, raw_indices_sorted, N, 0, sizeof(float)*8);
	cudaMalloc(&d_temp_storage_2, temp_storage_bytes_2);
	/**************************/
	
	int curr_print_idx = 0;
	float tax_dw = lambda*W_TOT/(p*N);
	for(int i = 0; i <= generations; ++i)
	{
		// printf("\rProgress %d%%", (i*100)/generations);

		/* evolucionar YSM */
		for(int j = 0; j < MCS_per_generation; ++j)
		{
			curandGenerateUniform(rng, raw_risk + N_TO_TRAIN, N-N_TO_TRAIN);

			curandGenerateUniform(rng, d_randoms, N);
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_randoms, d_randoms_sorted, raw_indices, raw_indices_shuffled, N, 0, sizeof(float)*8);
			trade_once<<<nBlocks_half, nThreads>>>(d_agents, raw_indices_shuffled, d_params, activ_fun, raw_p_rep, states);

			cub::DeviceRadixSort::SortPairs(d_temp_storage_2, temp_storage_bytes_2, raw_wealth, raw_wealth_sorted, raw_indices, raw_indices_sorted, N, 0, sizeof(float)*8);
			tax_the_ppl<<<nBlocks_all, nThreads>>>(raw_wealth, lambda);
			distribute_taxes<<<nBlocks_tax, nThreads>>>(tax_dw, p*N, raw_indices_sorted, raw_wealth);			
		}
		/*******************/			

		if(calc_gini && i%gini_generations == 0)
			print_gini(raw_wealth, i, filename_gini);

		/* producir nueva generacion */
		set_reproduction_probs(raw_p_rep);
		while(d_idx_new_last[0] < N_TO_TRAIN)
			kernel_reproduce<<<nBlocks_reproduce, nThreads>>>(raw_idx_new_last, raw_p_rep, d_params, d_params_new, mutagene, states);

		d_idx_new_last[0] = 0; //reinicio indice para la prox

		thrust::copy(d_biases_new.begin(), d_biases_new.end(), d_biases.begin());
		thrust::copy(d_weights_new.begin(), d_weights_new.end(), d_weights.begin());
		/*****************************/

		/* reset r, w y p_rep */
		curandGenerateUniform(rng, raw_risk, N);
		curandGenerateUniform(rng, raw_wealth, N);
		float sum_w_syst = thrust::reduce(d_wealth.begin(), d_wealth.end());
		thrust::transform(d_wealth.begin(), d_wealth.end(), d_wealth.begin(), _1/sum_w_syst);
		thrust::fill(d_p_rep.begin(), d_p_rep.end(), 0);
		/**********************/

		/* imprimir NNs */
		if(i == brains_generations_to_print[curr_print_idx])
		{
			h_biases = d_biases;
			h_weights = d_weights;

			string filename_brains_out = get_full_filename(layer_sizes, i, lambda, p, mutagene);
			print_brain_neurons(layer_sizes, h_biases, h_weights, filename_brains_out);

			++curr_print_idx;
		}
		/****************/
	}

	return 0;
}

__device__ float d_sigm(float x) { return 1/(1+expf(-x)); }
__device__ float d_ReLU(float x) { return (x < 0) ? 0 : x; }
__device__ float d_LeakyReLU(float x) { return (x < 0) ? 0.1*x : x; }

__device__ fun_ptr_t psigm = d_sigm;
__device__ fun_ptr_t pReLU = d_ReLU;
__device__ fun_ptr_t pLeakyReLU = d_LeakyReLU;

int main()
{
	/* function pointer stuff */
	fun_ptr_t sigm;
	fun_ptr_t ReLU;
	fun_ptr_t LeakyReLU;

	cudaMemcpyFromSymbol(&sigm, psigm, sizeof(fun_ptr_t));
	cudaMemcpyFromSymbol(&ReLU, pReLU, sizeof(fun_ptr_t));
	cudaMemcpyFromSymbol(&LeakyReLU, pLeakyReLU, sizeof(fun_ptr_t));
	/**************************/
		
	clock_t tic = clock();

	unsigned layer_sizes[] = {3};
	unsigned brains_generations_to_print[] = {10000, 50000, 100000, 200000};

	/***sample_agents_brains(unsigned* layer_sizes, int MCS_per_generation, int generations, float lambda, float p, 
							 float mutagene, fun_ptr_t activ_fun, unsigned* brains_generations_to_print,
							 bool load_brains, string filename_brains_in, 
						     bool calc_gini, int gini_generations, string filename_gini)***/
	sample_agents_brains(layer_sizes, 1000, 200000, 0.28, 0.52, 
						 0.05, sigm, brains_generations_to_print, 
						 false, "3n_100000g_10000t-90000rc_lamb0.28p0.52_mg0.05.txt", 
		                 false, 100, "gini_wonly_p0.1.txt");

	clock_t toc = clock();
	double time_ms_cpu = (toc - tic) / CLOCKS_PER_SEC;
	printf("Time: %f s\n", time_ms_cpu);
	
    return 0;
}