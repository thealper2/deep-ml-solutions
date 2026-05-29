import math

def analyze_array_performance(n: int, operations: list) -> dict:
    """
    Analyze and compare array operation performance across Python, Cython, and NumPy.
    
    Args:
        n: Array size (number of elements)
        operations: List of dicts with performance parameters for each operation
    
    Returns:
        dict with per-operation metrics, pipeline totals, recommendation,
        and crossover sizes.
    """
    results = []
    total_python = 0.0
    total_cython = 0.0
    total_numpy = 0.0
    crossover_sizes = {}
    
    for op in operations:
        name = op['name']
        
        python_time = (op['python_overhead_ns'] + n * op['python_per_elem_ns']) / 1000.0
        cython_time = (op['cython_overhead_ns'] + n * op['cython_per_elem_ns']) / 1000.0
        numpy_time = (op['numpy_overhead_ns'] + n * op['numpy_per_elem_ns']) / 1000.0
        
        cython_speedup = python_time / cython_time if cython_time > 0 else float('inf')
        numpy_speedup = python_time / numpy_time if numpy_time > 0 else float('inf')
        
        times = [('cython', cython_time), ('numpy', numpy_time), ('python', python_time)]
        times.sort(key=lambda x: (x[1], x[0]))
        best = times[0][0]
        
        results.append({
            'name': name,
            'python_us': round(python_time, 2),
            'cython_us': round(cython_time, 2),
            'numpy_us': round(numpy_time, 2),
            'cython_speedup': round(cython_speedup, 2),
            'numpy_speedup': round(numpy_speedup, 2),
            'best': best
        })
        
        total_python += python_time
        total_cython += cython_time
        total_numpy += numpy_time
        
        if op['cython_per_elem_ns'] <= op['numpy_per_elem_ns']:
            crossover_sizes[name] = -1
        else:
            numerator = op['numpy_overhead_ns'] - op['cython_overhead_ns']
            denominator = op['cython_per_elem_ns'] - op['numpy_per_elem_ns']
            if numerator <= 0:
                crossover_sizes[name] = 0
            else:
                crossover = math.ceil(numerator / denominator)
                crossover_sizes[name] = crossover
    
    pipeline_times = [
        ('cython', total_cython),
        ('numpy', total_numpy),
        ('python', total_python)
    ]
    pipeline_times.sort(key=lambda x: (x[1], x[0]))
    recommended = pipeline_times[0][0]
    
    return {
        'operations': results,
        'pipeline_python_us': round(total_python, 2),
        'pipeline_cython_us': round(total_cython, 2),
        'pipeline_numpy_us': round(total_numpy, 2),
        'recommended': recommended,
        'crossover_sizes': crossover_sizes
    }