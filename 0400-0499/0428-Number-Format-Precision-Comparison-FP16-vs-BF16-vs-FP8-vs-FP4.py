import numpy as np

def compare_formats(values):
    """
    Quantize values to FP16, BF16, FP8_E4M3, and FP4_E2M1 formats.
    """
    def fp16_quantize(vals):
        return _quantize_ieee(vals, exp_bits=5, mantissa_bits=10, has_inf=True)
    
    def bf16_quantize(vals):
        return _quantize_ieee(vals, exp_bits=8, mantissa_bits=7, has_inf=True)
    
    def fp8_e4m3_quantize(vals):
        return _quantize_ieee(vals, exp_bits=4, mantissa_bits=3, has_inf=False)
    
    def fp4_e2m1_quantize(vals):
        fp4_pos = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        fp4_neg = [-v for v in fp4_pos if v != 0]
        
        quantized = []
        for val in vals:
            if val == 0:
                quantized.append(0.0)
                continue
            sign = -1 if val < 0 else 1
            abs_val = abs(val)
            if abs_val > 4.0:
                quantized.append(sign * 4.0)
                continue
            candidates = fp4_pos if val >= 0 else fp4_neg
            best = candidates[0]
            best_dist = abs(abs_val - best)
            for c in candidates[1:]:
                dist = abs(abs_val - c)
                if dist < best_dist:
                    best = c
                    best_dist = dist
            quantized.append(sign * best)
        
        max_error = max(abs(v - q) for v, q in zip(vals, quantized) if np.isfinite(q))
        sum_error = sum(abs(v - q) for v, q in zip(vals, quantized) if np.isfinite(q))
        
        return {
            'max_representable': 4.0,
            'min_positive_normal': 1.0,
            'quantized': quantized,
            'max_abs_error': max_error,
            'mean_abs_error': sum_error / len(vals)
        }
    
    def _quantize_ieee(vals, exp_bits, mantissa_bits, has_inf):
        exp_bias = 2 ** (exp_bits - 1) - 1
        max_exp = 2 ** exp_bits - 1
        max_mantissa = 2 ** mantissa_bits
        mantissa_step = 1.0 / max_mantissa
        
        if has_inf:
            max_representable = (2.0 - mantissa_step) * (2.0 ** (max_exp - 1 - exp_bias))
        else:
            max_representable = (1.0 + (max_mantissa - 2) * mantissa_step) * (2.0 ** (max_exp - exp_bias))
        
        min_normal = 2.0 ** (1 - exp_bias)
        
        quantized = []
        for val in vals:
            if val == 0:
                quantized.append(0.0)
                continue
            
            sign = -1 if val < 0 else 1
            abs_val = abs(val)
            
            if abs_val > max_representable:
                quantized.append(max_representable * sign if not has_inf else float('inf') * sign)
                continue
            
            if abs_val < min_normal:
                quantized.append(0.0)
                continue
            
            exp = int(np.floor(np.log2(abs_val)))
            mantissa = abs_val / (2.0 ** exp)
            if mantissa >= 2.0:
                mantissa = 1.0
                exp += 1
            
            biased_exp = exp + exp_bias
            if biased_exp >= max_exp:
                quantized.append(max_representable * sign if not has_inf else float('inf') * sign)
                continue
            
            rounded_mantissa = round((mantissa - 1.0) * max_mantissa)
            rounded_mantissa = max(0, min(max_mantissa - 1, rounded_mantissa))
            
            quantized_mantissa = 1.0 + rounded_mantissa * mantissa_step
            q = quantized_mantissa * (2.0 ** exp) * sign
            
            if biased_exp == max_exp and rounded_mantissa == max_mantissa - 1:
                q = float('nan')
            quantized.append(q)
        
        errors = [abs(v - q) if np.isfinite(q) else float('inf') for v, q in zip(vals, quantized)]
        finite_errors = [e for e in errors if np.isfinite(e)]
        
        return {
            'max_representable': float(max_representable),
            'min_positive_normal': float(min_normal),
            'quantized': [round(q, 7) if isinstance(q, float) and not np.isnan(q) and not np.isinf(q) else q for q in quantized],
            'max_abs_error': max(finite_errors) if finite_errors else 0.0,
            'mean_abs_error': sum(finite_errors) / len(vals)
        }
    
    def compute_results(vals):
        return {
            'fp16': fp16_quantize(vals),
            'bf16': bf16_quantize(vals),
            'fp8_e4m3': fp8_e4m3_quantize(vals),
            'fp4_e2m1': fp4_e2m1_quantize(vals)
        }
    
    def format_results(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = format_results(v)
            elif isinstance(v, list):
                result[k] = v
            elif isinstance(v, float) and not np.isnan(v) and not np.isinf(v):
                if k in ['max_abs_error', 'mean_abs_error']:
                    result[k] = round(v, 6)
                else:
                    result[k] = v
            else:
                result[k] = v
        return result
    
    if hasattr(values, 'tolist'):
        values = values.tolist()
    
    result = compute_results(values)
    return format_results(result)
